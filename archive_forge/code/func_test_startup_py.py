import shutil
import sys
import tempfile
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
import pytest
from IPython.core.profileapp import list_bundled_profiles, list_profiles_in
from IPython.core.profiledir import ProfileDir
from IPython.testing import decorators as dec
from IPython.testing import tools as tt
from IPython.utils.process import getoutput
def test_startup_py(self):
    self.init('00-start.py', 'zzz=123\n', 'print(zzz)\n')
    self.validate('123')