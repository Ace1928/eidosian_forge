from distutils import errors
import imp
import os
import re
import shlex
import sys
import traceback
from setuptools.command import test
Run a module as a test module given its path.

    Args:
      module_path: The path to the module to test; must end in '.py'.

    Returns:
      True if the tests in this module pass, False if not or if an error occurs.
    