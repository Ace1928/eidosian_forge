from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import exceptions as http_exceptions
from googlecloudsdk.api_lib.functions import cmek_util
from googlecloudsdk.api_lib.functions.v1 import util as api_util
from googlecloudsdk.command_lib.functions import source_util
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files as file_utils
Add sources to function.

  Args:
    function: The function to add a source to.
    function_ref: The reference to the function.
    source_arg: Location of source code to deploy.
    stage_bucket: The name of the Google Cloud Storage bucket where source code
      will be stored.
    ignore_file: custom ignore_file name. Override .gcloudignore file to
      customize files to be skipped.
    kms_key: KMS key configured for the function.

  Returns:
    A list of fields on the function that have been changed.
  Raises:
    FunctionsError: If the kms_key doesn't exist or GCF P4SA lacks permissions.
  