import sys, platform, re, pytest
from numpy.core._multiarray_umath import (
import numpy as np
import subprocess
import pathlib
import os
import re
def _expect_error(self, msg, err_type, no_error_msg='Failed to generate error'):
    try:
        self._run()
    except subprocess.CalledProcessError as e:
        assertion_message = f'Expected: {msg}\nGot: {e.stderr}'
        assert re.search(msg, e.stderr), assertion_message
        assertion_message = f'Expected error of type: {err_type}; see full error:\n{e.stderr}'
        assert re.search(err_type, e.stderr), assertion_message
    else:
        assert False, no_error_msg