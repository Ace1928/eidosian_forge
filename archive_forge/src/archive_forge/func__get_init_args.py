import copy
import inspect
import pickle
import types
from dataclasses import fields, is_dataclass
from typing import Any, Dict, List, Literal, MutableMapping, Optional, Sequence, Tuple, Type, Union
from torch import nn
import pytorch_lightning as pl
from lightning_fabric.utilities.data import AttributeDict as _AttributeDict
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
def _get_init_args(frame: types.FrameType) -> Tuple[Optional[Any], Dict[str, Any]]:
    _, _, _, local_vars = inspect.getargvalues(frame)
    if '__class__' not in local_vars:
        return (None, {})
    cls = local_vars['__class__']
    init_parameters = inspect.signature(cls.__init__).parameters
    self_var, args_var, kwargs_var = parse_class_init_keys(cls)
    filtered_vars = [n for n in (self_var, args_var, kwargs_var) if n]
    exclude_argnames = (*filtered_vars, '__class__', 'frame', 'frame_args')
    local_args = {k: local_vars[k] for k in init_parameters}
    if kwargs_var:
        local_args.update(local_args.get(kwargs_var, {}))
    local_args = {k: v for k, v in local_args.items() if k not in exclude_argnames}
    self_arg = local_vars.get(self_var, None)
    return (self_arg, local_args)