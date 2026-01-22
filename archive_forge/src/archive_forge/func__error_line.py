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
def _error_line(error, obj, refimported):
    import traceback
    line = traceback.format_exc().splitlines()[-2].replace('[obj]', '[' + repr(obj) + ']')
    return 'while testing (with refimported=%s):  %s' % (refimported, line.lstrip())