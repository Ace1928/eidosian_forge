import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from packaging import version
from transformers.utils import is_tf_available
from ...utils import (
from ...utils.normalized_config import NormalizedConfigManager
from .base import ConfigBehavior, OnnxConfig, OnnxConfigWithPast, OnnxSeq2SeqConfigWithPast
from .config import (
from .model_patcher import (
class Pix2StructOnnxConfig(OnnxSeq2SeqConfigWithPast):
    NORMALIZED_CONFIG_CLASS = Pix2StructNormalizedConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, DummySeq2SeqDecoderTextInputGenerator, DummySeq2SeqPastKeyValuesGenerator, DummyPix2StructInputGenerator)
    DEFAULT_ONNX_OPSET = 12

    @property
    def inputs(self):
        common_inputs = {}
        common_inputs['attention_mask'] = {0: 'batch_size'}
        if self._behavior is not ConfigBehavior.DECODER:
            common_inputs['flattened_patches'] = {0: 'batch_size'}
        if self._behavior is not ConfigBehavior.ENCODER:
            if self.use_past_in_inputs:
                common_inputs['decoder_input_ids'] = {0: 'batch_size'}
            else:
                common_inputs['decoder_input_ids'] = {0: 'batch_size', 1: 'decoder_sequence_length'}
        if self._behavior is ConfigBehavior.DECODER:
            if self.use_past_in_inputs:
                self.add_past_key_values(common_inputs, direction='inputs')
            common_inputs['encoder_outputs'] = {0: 'batch_size'}
            common_inputs['decoder_attention_mask'] = {0: 'batch_size', 1: 'past_sequence_length + 1'}
        return common_inputs

    @property
    def outputs(self) -> Dict[str, Dict[int, str]]:
        if self._behavior is ConfigBehavior.ENCODER:
            common_outputs = {'last_hidden_state': {0: 'batch_size'}}
        else:
            common_outputs = super(OnnxConfigWithPast, self).outputs
        for name, axes_names in common_outputs.items():
            if self._behavior is ConfigBehavior.ENCODER or 'encoder' in name:
                sequence_name = 'encoder_sequence_length'
            else:
                sequence_name = 'decoder_sequence_length'
            new_axes_names = {}
            for axis_idx, axis_name in axes_names.items():
                if 'sequence' in axis_name:
                    if self.use_past_in_inputs is False or self.is_merged is True:
                        new_axes_names[axis_idx] = sequence_name
                    else:
                        new_axes_names[axis_idx] = '1'
                else:
                    new_axes_names[axis_idx] = axis_name
            common_outputs[name] = new_axes_names
        if self.use_past:
            self.add_past_key_values(common_outputs, direction='outputs')
        return common_outputs

    @property
    def torch_to_onnx_input_map(self) -> Dict[str, str]:
        if self._behavior is ConfigBehavior.DECODER:
            return {'decoder_input_ids': 'input_ids', 'encoder_outputs': 'encoder_hidden_states', 'attention_mask': 'encoder_attention_mask'}
        return {}

    def generate_dummy_inputs_for_validation(self, reference_model_inputs: Dict[str, Any], onnx_input_names: Optional[List[str]]=None) -> Dict[str, Any]:
        if self._behavior is ConfigBehavior.DECODER:
            reference_model_inputs['input_ids'] = reference_model_inputs.pop('decoder_input_ids')
        if onnx_input_names is not None:
            if 'encoder_outputs' in reference_model_inputs:
                if 'encoder_hidden_states' in onnx_input_names:
                    reference_model_inputs['encoder_hidden_states'] = reference_model_inputs.pop('encoder_outputs')[0]
                else:
                    reference_model_inputs.pop('encoder_outputs')
        else:
            reference_model_inputs['encoder_hidden_states'] = reference_model_inputs.pop('encoder_outputs')[0]
        return super().generate_dummy_inputs_for_validation(reference_model_inputs)

    def _create_dummy_input_generator_classes(self, **kwargs) -> List['DummyInputGenerator']:
        dummy_inputs_generators = []
        dummy_inputs_generators.append(self.DUMMY_INPUT_GENERATOR_CLASSES[0](self.task, self._normalized_config))
        if self._preprocessors is None or len(self._preprocessors) != 2:
            raise ValueError(f'Preprocessors for pix2struct need to be available for the ONNX export to infer input static shapes. Got: {self._preprocessors}')
        encoder_sequence_length = self._preprocessors[1].image_processor.max_patches
        kwargs['preprocessors'] = self._preprocessors
        for cls_ in self.DUMMY_INPUT_GENERATOR_CLASSES[1:]:
            dummy_inputs_generators.append(cls_(self.task, self._normalized_config, encoder_sequence_length=encoder_sequence_length, **kwargs))
        return dummy_inputs_generators

    def overwrite_shape_and_generate_input(self, dummy_input_gen: 'DummyInputGenerator', input_name: str, framework: str, input_shapes: Dict):
        if self._preprocessors is None or len(self._preprocessors) != 2:
            raise ValueError(f'Preprocessors for pix2struct need to be available for the ONNX export to infer input static shapes. Got: {self._preprocessors}')
        if self.use_past and self.use_past_in_inputs and (self.use_cache_branch is not False) and (input_name in ['decoder_input_ids', 'input_ids']):
            sequence_length = dummy_input_gen.sequence_length
            dummy_input_gen.sequence_length = 1
            dummy_input = dummy_input_gen.generate(input_name, framework=framework, int_dtype=self.int_dtype, float_dtype=self.float_dtype)
            dummy_input_gen.sequence_length = sequence_length
        elif input_name in ['encoder_outputs', 'attention_mask']:
            original_seq_length = dummy_input_gen.sequence_length
            dummy_input_gen.sequence_length = self._preprocessors[1].image_processor.max_patches
            dummy_input = dummy_input_gen.generate(input_name, framework=framework, int_dtype=self.int_dtype, float_dtype=self.float_dtype)
            dummy_input_gen.sequence_length = original_seq_length
        else:
            dummy_input = dummy_input_gen.generate(input_name, framework=framework, int_dtype=self.int_dtype, float_dtype=self.float_dtype)
        return dummy_input