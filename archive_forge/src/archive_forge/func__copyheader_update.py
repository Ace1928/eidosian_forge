import os
from warnings import warn
from ..base import traits, isdefined, TraitedSpec, File, Str, InputMultiObject
from ..mixins import CopyHeaderInterface
from .base import ANTSCommandInputSpec, ANTSCommand
def _copyheader_update(self):
    if self.inputs.copy_header and self.inputs.operation in self._no_copy_header_operation:
        warn(f'copy_header cannot be updated to True with {self.inputs.operation} as operation.')
        self.inputs.copy_header = False