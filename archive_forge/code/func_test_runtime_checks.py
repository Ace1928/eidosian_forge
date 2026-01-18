import os
import simplejson as json
import logging
import pytest
from unittest import mock
from .... import config
from ....testing import example_data
from ... import base as nib
from ..support import _inputs_help
def test_runtime_checks():

    class TestInterface(nib.BaseInterface):

        class input_spec(nib.TraitedSpec):
            a = nib.traits.Any()

        class output_spec(nib.TraitedSpec):
            b = nib.traits.Any()

        def _run_interface(self, runtime):
            return runtime

    class NoRuntime(TestInterface):

        def _run_interface(self, runtime):
            return None

    class BrokenRuntime(TestInterface):

        def _run_interface(self, runtime):
            del runtime.__dict__['cwd']
            return runtime
    with pytest.raises(RuntimeError):
        NoRuntime().run()
    with pytest.raises(RuntimeError):
        BrokenRuntime().run()