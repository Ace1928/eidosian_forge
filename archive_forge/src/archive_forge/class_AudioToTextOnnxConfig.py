from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union
from transformers.utils import is_tf_available
from ...onnx import merge_decoders
from ...utils import (
from .base import ConfigBehavior, OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
from .constants import ONNX_DECODER_MERGED_NAME, ONNX_DECODER_NAME, ONNX_DECODER_WITH_PAST_NAME
from .model_patcher import DecoderModelPatcher
class AudioToTextOnnxConfig(OnnxSeq2SeqConfigWithPast):
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyAudioInputGenerator, DummySeq2SeqDecoderTextInputGenerator, DummySeq2SeqPastKeyValuesGenerator)

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        common_inputs = {}
        if self._behavior is not ConfigBehavior.DECODER:
            common_inputs['input_features'] = {0: 'batch_size', 1: 'feature_size', 2: 'encoder_sequence_length'}
        if self._behavior is not ConfigBehavior.ENCODER:
            if self.use_past_in_inputs:
                common_inputs['decoder_input_ids'] = {0: 'batch_size'}
            else:
                common_inputs['decoder_input_ids'] = {0: 'batch_size', 1: 'decoder_sequence_length'}
            if self.use_past_in_inputs:
                self.add_past_key_values(common_inputs, direction='inputs')
        if self._behavior is ConfigBehavior.DECODER:
            common_inputs['encoder_outputs'] = {0: 'batch_size', 1: 'encoder_sequence_length'}
        return common_inputs

    @property
    def torch_to_onnx_input_map(self) -> Dict[str, str]:
        if self._behavior is ConfigBehavior.DECODER:
            return {'decoder_input_ids': 'input_ids', 'encoder_outputs': 'encoder_hidden_states', 'attention_mask': 'encoder_attention_mask'}
        return {}