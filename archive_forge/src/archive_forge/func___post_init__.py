from dataclasses import asdict, dataclass
from typing import Optional
from ...configuration_utils import PretrainedConfig
from ...utils import logging
def __post_init__(self):
    if self.structure_module is None:
        self.structure_module = StructureModuleConfig()
    elif isinstance(self.structure_module, dict):
        self.structure_module = StructureModuleConfig(**self.structure_module)
    if self.max_recycles <= 0:
        raise ValueError(f'`max_recycles` should be positive, got {self.max_recycles}.')
    if self.sequence_state_dim % self.sequence_state_dim != 0:
        raise ValueError(f'`sequence_state_dim` should be a round multiple of `sequence_state_dim`, got {self.sequence_state_dim} and {self.sequence_state_dim}.')
    if self.pairwise_state_dim % self.pairwise_state_dim != 0:
        raise ValueError(f'`pairwise_state_dim` should be a round multiple of `pairwise_state_dim`, got {self.pairwise_state_dim} and {self.pairwise_state_dim}.')
    sequence_num_heads = self.sequence_state_dim // self.sequence_head_width
    pairwise_num_heads = self.pairwise_state_dim // self.pairwise_head_width
    if self.sequence_state_dim != sequence_num_heads * self.sequence_head_width:
        raise ValueError(f'`sequence_state_dim` should be equal to `sequence_num_heads * sequence_head_width, got {self.sequence_state_dim} != {sequence_num_heads} * {self.sequence_head_width}.')
    if self.pairwise_state_dim != pairwise_num_heads * self.pairwise_head_width:
        raise ValueError(f'`pairwise_state_dim` should be equal to `pairwise_num_heads * pairwise_head_width, got {self.pairwise_state_dim} != {pairwise_num_heads} * {self.pairwise_head_width}.')
    if self.pairwise_state_dim % 2 != 0:
        raise ValueError(f'`pairwise_state_dim` should be even, got {self.pairwise_state_dim}.')
    if self.dropout >= 0.4:
        raise ValueError(f'`dropout` should not be greater than 0.4, got {self.dropout}.')