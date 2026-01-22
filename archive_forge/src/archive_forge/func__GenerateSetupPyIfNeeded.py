from the command line arguments and returns a list of URLs to be given to the
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import contextlib
import io
import os
import sys
import textwrap
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.command_lib.ml_engine import uploads
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
import six
from six.moves import map
from setuptools import setup, find_packages
def _GenerateSetupPyIfNeeded(setup_py_path, package_name):
    """Generates a temporary setup.py file if there is none at the given path.

  Args:
    setup_py_path: str, a path to the expected setup.py location.
    package_name: str, the name of the Python package for which to write a
      setup.py file (used in the generated file contents).

  Returns:
    bool, whether the setup.py file was generated.
  """
    log.debug('Looking for setup.py file at [%s]', setup_py_path)
    if os.path.isfile(setup_py_path):
        log.info('Using existing setup.py file at [%s]', setup_py_path)
        return False
    setup_contents = DEFAULT_SETUP_FILE.format(package_name=package_name)
    log.info('Generating temporary setup.py file:\n%s', setup_contents)
    files.WriteFileContents(setup_py_path, setup_contents)
    return True