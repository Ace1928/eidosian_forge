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
class SetuptoolsFailedError(UploadFailureError):
    """Error indicating that setuptools itself failed."""

    def __init__(self, output, generated):
        msg = 'Packaging of user Python code failed with message:\n\n{}\n\n'.format(output)
        if generated:
            msg += 'Try manually writing a setup.py file at your package root and rerunning the command.'
        else:
            msg += 'Try manually building your Python code by running:\n  $ python setup.py sdist\nand providing the output via the `--packages` flag (for example, `--packages dist/package.tar.gz,dist/package2.whl)`'
        super(SetuptoolsFailedError, self).__init__(msg)