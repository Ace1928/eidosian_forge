import abc
import torch
from typing import Optional, Tuple, List, Any, Dict
from ...sparsifier import base_sparsifier
from collections import defaultdict
from torch import nn
import copy
from ...sparsifier import utils
from torch.nn.utils import parametrize
import sys
import warnings
def _load_container_from_state(self, states, data_groups, container_state_dict):
    """This restores the state of the container specifically based on the data present in state and data_groups
        If the data was parametrized, then the data would be added to the container and then parametrized,
        else it would just add the attribute the container.
        """
    for name, state in states.items():
        config_name = data_groups.get(name, None)
        if config_name is None:
            raise RuntimeError(f'Error loading {name}')
        parametrized_name = f'parametrizations.{name}.original'
        parametrized = False
        data = container_state_dict.get(name, None)
        if name in container_state_dict:
            data = container_state_dict.get(name)
        elif parametrized_name in container_state_dict:
            data = container_state_dict.get(parametrized_name)
            parametrized = True
        else:
            raise RuntimeError(f'Error loading {name}')
        self._container.register_buffer(name=name, tensor=data)
        if parametrized:
            mask = state.get('mask', torch.ones_like(data))
            param_class = data_groups.get('parametrization', utils.FakeSparsity)
            parametrize.register_parametrization(self._container, name, param_class(mask))