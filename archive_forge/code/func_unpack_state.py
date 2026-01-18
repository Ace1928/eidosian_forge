from collections import OrderedDict
import gymnasium as gym
from typing import Union, Dict, List, Tuple
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType
def unpack_state(self, state: List[TensorType]) -> Tuple[List[Tuple[TensorType, TensorType]], Dict[str, TensorType], TensorType]:
    """Given a list of tensors, reformat for self.dnc input"""
    assert len(state) == 9, 'Failed to verify unpacked state'
    ctrl_hidden: List[Tuple[TensorType, TensorType]] = [(state[0].permute(1, 0, 2).contiguous(), state[1].permute(1, 0, 2).contiguous())]
    read_vecs: TensorType = state[2]
    memory: List[TensorType] = state[3:]
    memory_dict: OrderedDict[str, TensorType] = OrderedDict(zip(self.MEMORY_KEYS, memory))
    return (ctrl_hidden, memory_dict, read_vecs)