import os
import simplejson as json
import logging
import pytest
from unittest import mock
from .... import config
from ....testing import example_data
from ... import base as nib
from ..support import _inputs_help
def test_unavailable_input():

    class WithInput(nib.BaseInterface):

        class input_spec(nib.TraitedSpec):
            foo = nib.traits.Int(3, usedefault=True, max_ver='0.5')
        _version = '0.4'

        def _run_interface(self, runtime):
            return runtime

    class WithoutInput(WithInput):
        _version = '0.6'
    has = WithInput()
    hasnt = WithoutInput()
    trying_anyway = WithoutInput(foo=3)
    assert has.inputs.foo == 3
    assert not nib.isdefined(hasnt.inputs.foo)
    assert trying_anyway.inputs.foo == 3
    has.run()
    hasnt.run()
    with pytest.raises(Exception):
        trying_anyway.run()
    has.inputs.foo = 4
    hasnt.inputs.foo = 4
    trying_anyway.inputs.foo = 4
    assert has.inputs.foo == 4
    assert hasnt.inputs.foo == 4
    assert trying_anyway.inputs.foo == 4
    has.run()
    with pytest.raises(Exception):
        hasnt.run()
    with pytest.raises(Exception):
        trying_anyway.run()