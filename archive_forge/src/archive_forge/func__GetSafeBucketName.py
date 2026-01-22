from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import random
import string
import time
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py.exceptions import HttpError
from apitools.base.py.exceptions import HttpNotFoundError
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.cloudbuild import logs as cb_logs
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.services import enable_api as services_api
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions as http_exc
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.artifacts import docker_util
from googlecloudsdk.command_lib.cloudbuild import execution
from googlecloudsdk.command_lib.compute.sole_tenancy import util as sole_tenancy_util
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import requests
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import encoding as encoding_util
import six
def _GetSafeBucketName(bucket_name, add_random_suffix=False):
    """Updates bucket name to meet https://cloud.google.com/storage/docs/naming.

  Args:
    bucket_name: str, input bucket name.
    add_random_suffix: bool, if specified a random suffix is added to its name.

  Returns:
    str, safe bucket name.
  """
    bucket_name = bucket_name.replace('google', 'go-ogle')
    if add_random_suffix:
        suffix = _GenerateRandomBucketSuffix()
        suffix = suffix.replace('google', 'go-ogle')
    else:
        suffix = ''
    bucket_name = bucket_name[:4].replace('goog', 'go-og') + bucket_name[4:]
    max_len = 63 - len(suffix)
    if len(bucket_name) > max_len:
        bucket_name = bucket_name[:max_len]
    return bucket_name + suffix