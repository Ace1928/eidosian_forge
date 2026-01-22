import os
from warnings import warn
from ..base import traits, isdefined, TraitedSpec, File, Str, InputMultiObject
from ..mixins import CopyHeaderInterface
from .base import ANTSCommandInputSpec, ANTSCommand
def _operation_update(self):
    if self.inputs.operation in self._no_copy_header_operation:
        self.inputs.copy_header = False