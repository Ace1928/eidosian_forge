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
def test_show_message_with_error(caplog):
    error_msg = "Doesn't work."

    def f(message):
        raise Exception(error_msg)
    with utils.obj_in_module(callbacks, 'showmessage', f), caplog.at_level(logging.ERROR, logger='callbacks.logger'):
        caplog.clear()
        msg = openrlib.ffi.new('char []', b'foo')
        callbacks._showmessage(msg)
        assert len(caplog.record_tuples) > 0
        for x in caplog.record_tuples:
            assert x == ('rpy2.rinterface_lib.callbacks', logging.ERROR, callbacks._SHOWMESSAGE_EXCEPTION_LOG % error_msg)