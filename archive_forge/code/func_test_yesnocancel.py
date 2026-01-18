from .. import utils
import builtins
import io
import logging
import os
import pytest
import tempfile
import sys
import rpy2.rinterface as rinterface
from rpy2.rinterface_lib import callbacks
from rpy2.rinterface_lib import openrlib
def test_yesnocancel():

    def yesnocancel(question):
        return 1
    question = openrlib.ffi.new('char []', b'What ?')
    with utils.obj_in_module(callbacks, 'yesnocancel', yesnocancel):
        res = callbacks._yesnocancel(question)
    assert res == 1