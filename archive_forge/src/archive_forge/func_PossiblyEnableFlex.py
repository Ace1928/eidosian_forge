from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import re
from apitools.base.py import exceptions as apitools_exceptions
from gae_ext_runtime import ext_runtime
from googlecloudsdk.api_lib.app import appengine_api_client
from googlecloudsdk.api_lib.app import build as app_build
from googlecloudsdk.api_lib.app import cloud_build
from googlecloudsdk.api_lib.app import docker_image
from googlecloudsdk.api_lib.app import metric_names
from googlecloudsdk.api_lib.app import runtime_builders
from googlecloudsdk.api_lib.app import util
from googlecloudsdk.api_lib.app import yaml_parsing
from googlecloudsdk.api_lib.app.images import config
from googlecloudsdk.api_lib.app.runtimes import fingerprinter
from googlecloudsdk.api_lib.cloudbuild import build as cloudbuild_build
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.services import exceptions as s_exceptions
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import exceptions as api_lib_exceptions
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.credentials import creds
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
from googlecloudsdk.third_party.appengine.api import appinfo
from googlecloudsdk.third_party.appengine.tools import context_util
import six
from six.moves import filter  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin
def PossiblyEnableFlex(project):
    """Attempts to enable the Flexible Environment API on the project.

  Possible scenarios:
  -If Flexible Environment is already enabled, success.
  -If Flexible Environment API is not yet enabled, attempts to enable it. If
   that succeeds, success.
  -If the account doesn't have permissions to confirm that the Flexible
   Environment API is or isn't enabled on this project, succeeds with a warning.
     -If the account is a service account, adds an additional warning that
      the Service Management API may need to be enabled.
  -If the Flexible Environment API is not enabled on the project and the attempt
   to enable it fails, raises PrepareFailureError.

  Args:
    project: str, the project ID.

  Raises:
    PrepareFailureError: if enabling the API fails with a 403 or 404 error code.
    googlecloudsdk.api_lib.util.exceptions.HttpException: miscellaneous errors
        returned by server.
  """
    try:
        enable_api.EnableServiceIfDisabled(project, 'appengineflex.googleapis.com')
    except s_exceptions.GetServicePermissionDeniedException:
        warning = FLEXIBLE_SERVICE_VERIFY_WARNING.format(project)
        credential = c_store.LoadIfEnabled(use_google_auth=True)
        if credential and creds.IsServiceAccountCredentials(credential):
            warning += '\n\n{}'.format(FLEXIBLE_SERVICE_VERIFY_WITH_SERVICE_ACCOUNT)
        log.warning(warning)
    except s_exceptions.EnableServicePermissionDeniedException:
        raise PrepareFailureError(PREPARE_FAILURE_MSG.format(project))
    except apitools_exceptions.HttpError as err:
        raise api_lib_exceptions.HttpException(err, error_format='Error [{status_code}] {status_message}{error.details?\nDetailed error information:\n{?}}')