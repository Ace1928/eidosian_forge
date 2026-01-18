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
def test_showmessage_default(capsys):
    buf = 'foo'
    callbacks.showmessage(buf)
    captured = capsys.readouterr()
    assert captured.out.split('\n')[1] == buf