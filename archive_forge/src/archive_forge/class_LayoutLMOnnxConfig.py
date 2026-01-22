from collections import OrderedDict
from typing import Any, List, Mapping, Optional
from ... import PretrainedConfig, PreTrainedTokenizer
from ...onnx import OnnxConfig, PatchingSpec
from ...utils import TensorType, is_torch_available, logging
class LayoutLMOnnxConfig(OnnxConfig):

    def __init__(self, config: PretrainedConfig, task: str='default', patching_specs: List[PatchingSpec]=None):
        super().__init__(config, task=task, patching_specs=patching_specs)
        self.max_2d_positions = config.max_2d_position_embeddings - 1

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict([('input_ids', {0: 'batch', 1: 'sequence'}), ('bbox', {0: 'batch', 1: 'sequence'}), ('attention_mask', {0: 'batch', 1: 'sequence'}), ('token_type_ids', {0: 'batch', 1: 'sequence'})])

    def generate_dummy_inputs(self, tokenizer: PreTrainedTokenizer, batch_size: int=-1, seq_length: int=-1, is_pair: bool=False, framework: Optional[TensorType]=None) -> Mapping[str, Any]:
        """
        Generate inputs to provide to the ONNX exporter for the specific framework

        Args:
            tokenizer: The tokenizer associated with this model configuration
            batch_size: The batch size (int) to export the model for (-1 means dynamic axis)
            seq_length: The sequence length (int) to export the model for (-1 means dynamic axis)
            is_pair: Indicate if the input is a pair (sentence 1, sentence 2)
            framework: The framework (optional) the tokenizer will generate tensor for

        Returns:
            Mapping[str, Tensor] holding the kwargs to provide to the model's forward function
        """
        input_dict = super().generate_dummy_inputs(tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework)
        box = [48, 84, 73, 128]
        if not framework == TensorType.PYTORCH:
            raise NotImplementedError('Exporting LayoutLM to ONNX is currently only supported for PyTorch.')
        if not is_torch_available():
            raise ValueError('Cannot generate dummy inputs without PyTorch installed.')
        import torch
        batch_size, seq_length = input_dict['input_ids'].shape
        input_dict['bbox'] = torch.tensor([*[box] * seq_length]).tile(batch_size, 1, 1)
        return input_dict