import pytest
from .... import config
from ... import engine as pe
from ....interfaces import base as nib
from ....interfaces.utility import IdentityInterface, Function, Merge
from ....interfaces.base import traits, File
class IncrementInterface(nib.SimpleInterface):
    input_spec = IncrementInputSpec
    output_spec = IncrementOutputSpec

    def _run_interface(self, runtime):
        runtime.returncode = 0
        self._results['output1'] = self.inputs.input1 + self.inputs.inc
        return runtime