import os
import simplejson as json
import logging
import pytest
from unittest import mock
from .... import config
from ....testing import example_data
from ... import base as nib
from ..support import _inputs_help
def test_output_version():

    class InputSpec(nib.TraitedSpec):
        foo = nib.traits.Int(desc='a random int')

    class OutputSpec(nib.TraitedSpec):
        foo = nib.traits.Int(desc='a random int', min_ver='0.9')

    class DerivedInterface1(nib.BaseInterface):
        input_spec = InputSpec
        output_spec = OutputSpec
        _version = '0.10'
        resource_monitor = False
    obj = DerivedInterface1()
    assert obj._check_version_requirements(obj._outputs()) == []

    class InputSpec(nib.TraitedSpec):
        foo = nib.traits.Int(desc='a random int')

    class OutputSpec(nib.TraitedSpec):
        foo = nib.traits.Int(desc='a random int', min_ver='0.11')

    class DerivedInterface1(nib.BaseInterface):
        input_spec = InputSpec
        output_spec = OutputSpec
        _version = '0.10'
        resource_monitor = False
    obj = DerivedInterface1()
    assert obj._check_version_requirements(obj._outputs()) == ['foo']

    class InputSpec(nib.TraitedSpec):
        foo = nib.traits.Int(desc='a random int')

    class OutputSpec(nib.TraitedSpec):
        foo = nib.traits.Int(desc='a random int', min_ver='0.11')

    class DerivedInterface1(nib.BaseInterface):
        input_spec = InputSpec
        output_spec = OutputSpec
        _version = '0.10'
        resource_monitor = False

        def _run_interface(self, runtime):
            return runtime

        def _list_outputs(self):
            return {'foo': 1}
    obj = DerivedInterface1()
    with pytest.raises(KeyError):
        obj.run()