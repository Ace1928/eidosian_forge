import inspect
from typing import Any, Dict, List, Optional, Union
import torch.nn
from . import utils, variables
from .bytecode_transformation import (
from .codegen import PyCodegen
from .exc import unimplemented
from .source import LocalSource, Source
from .utils import nn_module_new, object_new
from .variables.base import (
def track_object_new(self, cls_source: Source, user_cls: Any, variable_cls: Any, options):
    if user_cls is torch.autograd.function.FunctionCtx:
        obj = torch.autograd.Function()
    elif issubclass(user_cls, torch.nn.Module):
        obj = nn_module_new(user_cls)
    else:
        obj = object_new(user_cls)
    variable = variable_cls(obj, mutable_local=AttributeMutationNew(None, cls_source), **options)
    self.id_to_variable[id(obj)] = variable
    self.keepalive.append(obj)
    return variable