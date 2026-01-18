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
def track_cell_new(self):
    obj = object()
    variable = variables.NewCellVariable(mutable_local=AttributeMutationNew(None, None))
    self.id_to_variable[id(obj)] = variable
    self.keepalive.append(obj)
    return variable