import json
import os
import warnings
from unittest import mock
import pytest
from IPython import display
from IPython.core.getipython import get_ipython
from IPython.utils.io import capture_output
from IPython.utils.tempdir import NamedFileInTemporaryDirectory
from IPython import paths as ipath
from IPython.testing.tools import AssertNotPrints
import IPython.testing.decorators as dec
def test_update_display():
    ip = get_ipython()
    with mock.patch.object(ip.display_pub, 'publish') as pub:
        with pytest.raises(TypeError):
            display.update_display('x')
        display.update_display('x', display_id='1')
        display.update_display('y', display_id='2')
    args, kwargs = pub.call_args_list[0]
    assert args == ()
    assert kwargs == {'data': {'text/plain': repr('x')}, 'metadata': {}, 'transient': {'display_id': '1'}, 'update': True}
    args, kwargs = pub.call_args_list[1]
    assert args == ()
    assert kwargs == {'data': {'text/plain': repr('y')}, 'metadata': {}, 'transient': {'display_id': '2'}, 'update': True}