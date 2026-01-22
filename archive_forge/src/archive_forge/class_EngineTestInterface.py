import pytest
from ..base import EngineBase
from ....interfaces import base as nib
from ....interfaces import utility as niu
from ... import engine as pe
class EngineTestInterface(nib.SimpleInterface):
    input_spec = InputSpec
    output_spec = OutputSpec

    def _run_interface(self, runtime):
        runtime.returncode = 0
        self._results['output1'] = [1, self.inputs.input1]
        return runtime