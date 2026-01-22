from __future__ import absolute_import, division, print_function
import collections
import sys
import time
import datetime
import os
import platform
import re
import functools
from contextlib import contextmanager
def _couldNotImportPyGetWindow(*unused_args, **unused_kwargs):
    """
            This function raises PyAutoGUIException. It's used for the PyGetWindow function names if the PyGetWindow
            module failed to be imported.
            """
    raise PyAutoGUIException('PyAutoGUI was unable to import pygetwindow. Please install this module to enable the function you tried to call.')