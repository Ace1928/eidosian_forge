import atexit
import os
import sys
import __main__
from contextlib import suppress
from io import BytesIO
import dill
import json                                         # top-level module
import urllib as url                                # top-level module under alias
from xml import sax                                 # submodule
import xml.dom.minidom as dom                       # submodule under alias
import test_dictviews as local_mod                  # non-builtin top-level module
from calendar import Calendar, isleap, day_name     # class, function, other object
from cmath import log as complex_log                # imported with alias
def test_session_main(refimported):
    """test dump/load_module() for __main__, both in this process and in a subprocess"""
    extra_objects = {}
    if refimported:
        from sys import flags
        extra_objects['flags'] = flags
    with TestNamespace(**extra_objects) as ns:
        try:
            dill.dump_module(session_file % refimported, refimported=refimported)
            from dill.tests.__main__ import python, shell, sp
            error = sp.call([python, __file__, '--child', str(refimported)], shell=shell)
            if error:
                sys.exit(error)
        finally:
            with suppress(OSError):
                os.remove(session_file % refimported)
        session_buffer = BytesIO()
        dill.dump_module(session_buffer, refimported=refimported)
        session_buffer.seek(0)
        dill.load_module(session_buffer, module='__main__')
        ns.backup['_test_objects'](__main__, ns.backup, refimported)