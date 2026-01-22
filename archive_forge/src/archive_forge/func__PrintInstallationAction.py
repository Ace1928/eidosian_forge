from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import functools
import sys
import traceback
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.error_reporting import util
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.calliope import command_loading
from googlecloudsdk.command_lib import error_reporting_util
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.util import platforms
def _PrintInstallationAction(err, err_string):
    """Prompts installation error action.

  Args:
    err: Exception err.
    err_string: Exception err string.
  """
    log.error('gcloud failed to load ({command}): {err_str}\n\nThis usually indicates corruption in your gcloud installation or problems with your Python interpreter.\n\nPlease verify that the following is the path to a working Python {py_major_version}.{py_minor_version}+ executable:\n    {executable}\nIf it is not, please set the CLOUDSDK_PYTHON environment variable to point to a working Python {py_major_version}.{py_minor_version}+ executable.\n\nIf you are still experiencing problems, please run the following command to reinstall:\n    $ gcloud components reinstall\n\nIf that command fails, please reinstall the Google Cloud CLI using the instructions here:\n    https://cloud.google.com/sdk/'.format(command=err.command, err_str=err_string, executable=sys.executable, py_major_version=platforms.PythonVersion.MIN_SUPPORTED_PY3_VERSION[0], py_minor_version=platforms.PythonVersion.MIN_SUPPORTED_PY3_VERSION[1]))