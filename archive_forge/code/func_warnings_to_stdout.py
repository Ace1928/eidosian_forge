import sys
import warnings
import os.path
import re
import subprocess
import threading
import pytest
import _pytest
def warnings_to_stdout():
    """ Redirect all warnings to stdout.
    """
    showwarning_orig = warnings.showwarning

    def showwarning(msg, cat, fname, lno, file=None, line=0):
        showwarning_orig(msg, cat, os.path.basename(fname), line, sys.stdout)
    warnings.showwarning = showwarning