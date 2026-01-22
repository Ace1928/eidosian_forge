from __future__ import annotations
import collections.abc
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
from .configuration_data2vec_vision import Data2VecVisionConfig
class TFData2VecVisionFCNHead(keras.layers.Layer):
    """
    Fully Convolution Networks for Semantic Segmentation. This head is implemented from
    [FCNNet](https://arxiv.org/abs/1411.4038).

    Args:
        config (Data2VecVisionConfig): Configuration.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        dilation (int): The dilation rate for convs in the head. Default: 1.


    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """

    def __init__(self, config: Data2VecVisionConfig, in_index: int=2, kernel_size: int=3, dilation: Union[int, Tuple[int, int]]=1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.in_channels = config.hidden_size
        self.channels = config.auxiliary_channels
        self.num_convs = config.auxiliary_num_convs
        self.concat_input = config.auxiliary_concat_input
        self.in_index = in_index
        convs = []
        convs.append(TFData2VecVisionConvModule(in_channels=self.in_channels, out_channels=self.channels, kernel_size=kernel_size, padding='same', dilation=dilation, name='convs.0'))
        for i in range(self.num_convs - 1):
            convs.append(TFData2VecVisionConvModule(in_channels=self.channels, out_channels=self.channels, kernel_size=kernel_size, padding='same', dilation=dilation, name=f'conv_module_{i + 2}'))
        if self.num_convs == 0:
            self.convs = [tf.identity]
        else:
            self.convs = convs
        if self.concat_input:
            self.conv_cat = TFData2VecVisionConvModule(self.in_channels + self.channels, out_channels=self.channels, kernel_size=kernel_size, padding='same', name='conv_cat')
        self.classifier = keras.layers.Conv2D(config.num_labels, kernel_size=1, name='classifier')

    def call(self, encoder_hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = encoder_hidden_states[self.in_index]
        output = hidden_states
        for layer_module in self.convs:
            output = layer_module(output)
        if self.concat_input:
            output = self.conv_cat(tf.concat([hidden_states, output], axis=-1))
        output = self.classifier(output)
        return output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'classifier', None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, None, self.channels])
        if getattr(self, 'conv_cat', None) is not None:
            with tf.name_scope(self.conv_cat.name):
                self.conv_cat.build(None)