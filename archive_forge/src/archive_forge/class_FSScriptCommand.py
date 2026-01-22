import os
from ... import LooseVersion
from ...utils.filemanip import fname_presuffix
from ..base import (
class FSScriptCommand(FSCommand):
    """Support for Freesurfer script commands with log terminal_output"""
    _terminal_output = 'file'
    _always_run = False

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['log_file'] = os.path.abspath('output.nipype')
        return outputs