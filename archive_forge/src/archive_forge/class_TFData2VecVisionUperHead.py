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
class TFData2VecVisionUperHead(keras.layers.Layer):
    """
    Unified Perceptual Parsing for Scene Understanding. This head is the implementation of
    [UPerNet](https://arxiv.org/abs/1807.10221).

    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """

    def __init__(self, config: Data2VecVisionConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.pool_scales = config.pool_scales
        self.in_channels = [config.hidden_size] * 4
        self.channels = config.hidden_size
        self.classifier = keras.layers.Conv2D(config.num_labels, kernel_size=1, name='classifier')
        self.psp_modules = TFData2VecVisionPyramidPoolingModule(self.pool_scales, self.in_channels[-1], self.channels, name='psp_modules')
        self.bottleneck = TFData2VecVisionConvModule(self.in_channels[-1] + len(self.pool_scales) * self.channels, self.channels, kernel_size=3, padding='same', name='bottleneck')
        self.lateral_convs = []
        self.fpn_convs = []
        for idx, in_channels in enumerate(self.in_channels[:-1]):
            l_conv = TFData2VecVisionConvModule(in_channels, out_channels=self.channels, kernel_size=1, name=f'lateral_convs.{idx}')
            fpn_conv = TFData2VecVisionConvModule(in_channels=self.channels, out_channels=self.channels, kernel_size=3, padding='same', name=f'fpn_convs.{idx}')
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        self.fpn_bottleneck = TFData2VecVisionConvModule(in_channels=len(self.in_channels) * self.channels, out_channels=self.channels, kernel_size=3, padding='same', name='fpn_bottleneck')

    def psp_forward(self, inputs):
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = tf.concat(psp_outs, axis=-1)
        output = self.bottleneck(psp_outs)
        return output

    def call(self, encoder_hidden_states: tf.Tensor) -> tf.Tensor:
        laterals = [lateral_conv(encoder_hidden_states[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        laterals.append(self.psp_forward(encoder_hidden_states))
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = shape_list(laterals[i - 1])[1:-1]
            laterals[i - 1] = laterals[i - 1] + tf.image.resize(laterals[i], size=prev_shape, method='bilinear')
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)]
        fpn_outs.append(laterals[-1])
        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = tf.image.resize(fpn_outs[i], size=shape_list(fpn_outs[0])[1:-1], method='bilinear')
        fpn_outs = tf.concat(fpn_outs, axis=-1)
        output = self.fpn_bottleneck(fpn_outs)
        output = self.classifier(output)
        return output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'classifier', None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, None, self.channels])
        if getattr(self, 'psp_modules', None) is not None:
            with tf.name_scope(self.psp_modules.name):
                self.psp_modules.build(None)
        if getattr(self, 'bottleneck', None) is not None:
            with tf.name_scope(self.bottleneck.name):
                self.bottleneck.build(None)
        if getattr(self, 'fpn_bottleneck', None) is not None:
            with tf.name_scope(self.fpn_bottleneck.name):
                self.fpn_bottleneck.build(None)
        for layer in self.lateral_convs:
            with tf.name_scope(layer.name):
                layer.build(None)
        for layer in self.fpn_convs:
            with tf.name_scope(layer.name):
                layer.build(None)