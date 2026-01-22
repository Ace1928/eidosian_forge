from __future__ import annotations
import math
import random
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...generation.configuration_utils import GenerationConfig
from ...generation.tf_logits_process import TFLogitsProcessorList
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_whisper import WhisperConfig
from .tokenization_whisper import TASK_IDS, TO_LANGUAGE_CODE
class TFWhisperPreTrainedModel(TFPreTrainedModel):
    config_class = WhisperConfig
    base_model_prefix = 'model'
    main_input_name = 'input_features'

    def _get_feat_extract_output_lengths(self, input_lengths: tf.Tensor) -> int:
        """
        Computes the output length of the convolutional layers
        """
        input_lengths = (input_lengths - 1) // 2 + 1
        return input_lengths

    @property
    def dummy_inputs(self) -> Dict[str, tf.Tensor]:
        """
        Dummy inputs to build the network.

        Returns:
            `Dict[str, tf.Tensor]`: The dummy inputs.
        """
        return {self.main_input_name: tf.random.uniform([1, self.config.num_mel_bins, self.config.max_source_positions * 2 - 1], dtype=tf.float32), 'decoder_input_ids': tf.constant([[1, 3]], dtype=tf.int32)}

    @property
    def input_signature(self):
        return {'input_features': tf.TensorSpec((None, self.config.num_mel_bins, None), tf.float32, name='input_features'), 'decoder_input_ids': tf.TensorSpec((None, None), tf.int32, name='decoder_input_ids'), 'decoder_attention_mask': tf.TensorSpec((None, None), tf.int32, name='decoder_attention_mask')}