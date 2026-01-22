import copy
import inspect
import types
from argparse import Namespace
from typing import Any, List, MutableMapping, Optional, Sequence, Union
from lightning_fabric.utilities.data import AttributeDict
from pytorch_lightning.utilities.parsing import save_hyperparameters
@staticmethod
def _to_hparams_dict(hp: Union[MutableMapping, Namespace, str]) -> Union[MutableMapping, AttributeDict]:
    if isinstance(hp, Namespace):
        hp = vars(hp)
    if isinstance(hp, dict):
        hp = AttributeDict(hp)
    elif isinstance(hp, _PRIMITIVE_TYPES):
        raise ValueError(f'Primitives {_PRIMITIVE_TYPES} are not allowed.')
    elif not isinstance(hp, _ALLOWED_CONFIG_TYPES):
        raise ValueError(f'Unsupported config type of {type(hp)}.')
    return hp