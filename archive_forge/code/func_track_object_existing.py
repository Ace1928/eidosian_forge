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
def track_object_existing(self, source: Source, item: Any, variable: VariableTracker):
    return self._track_obj(source, item, variable, mutable_cls=AttributeMutationExisting)