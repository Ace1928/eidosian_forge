import pytest
from .... import config
from ... import engine as pe
from ....interfaces import base as nib
from ....interfaces.utility import IdentityInterface, Function, Merge
from ....interfaces.base import traits, File
class SetInterface(nib.BaseInterface):
    input_spec = SetInputSpec
    output_spec = SetOutputSpec

    def _run_interface(self, runtime):
        runtime.returncode = 0
        return runtime

    def _list_outputs(self):
        global _set_len
        outputs = self._outputs().get()
        _set_len = outputs['output1'] = len(self.inputs.input1)
        return outputs