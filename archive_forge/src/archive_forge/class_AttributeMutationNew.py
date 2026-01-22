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
class AttributeMutationNew(AttributeMutation):

    def __init__(self, source: Optional[Source], cls_source: Optional[Source]):
        super().__init__(MutableLocalSource.Local, source)
        self.cls_source = cls_source