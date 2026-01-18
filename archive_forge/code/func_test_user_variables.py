import asyncio
import ast
import os
import signal
import shutil
import sys
import tempfile
import unittest
import pytest
from unittest import mock
from os.path import join
from IPython.core.error import InputRejected
from IPython.core.inputtransformer import InputTransformer
from IPython.core import interactiveshell
from IPython.core.oinspect import OInfo
from IPython.testing.decorators import (
from IPython.testing import tools as tt
from IPython.utils.process import find_cmd
import warnings
import warnings
def test_user_variables():
    ip.display_formatter.active_types = ip.display_formatter.format_types
    ip.user_ns['dummy'] = d = DummyRepr()
    keys = {'dummy', 'doesnotexist'}
    r = ip.user_expressions({key: key for key in keys})
    assert keys == set(r.keys())
    dummy = r['dummy']
    assert {'status', 'data', 'metadata'} == set(dummy.keys())
    assert dummy['status'] == 'ok'
    data = dummy['data']
    metadata = dummy['metadata']
    assert data.get('text/html') == d._repr_html_()
    js, jsmd = d._repr_javascript_()
    assert data.get('application/javascript') == js
    assert metadata.get('application/javascript') == jsmd
    dne = r['doesnotexist']
    assert dne['status'] == 'error'
    assert dne['ename'] == 'NameError'
    ip.display_formatter.active_types = ['text/plain']