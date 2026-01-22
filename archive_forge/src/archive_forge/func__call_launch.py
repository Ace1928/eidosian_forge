from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import http_wrapper as apitools_http_wrapper
from googlecloudsdk.api_lib.storage import errors as cloud_errors
from googlecloudsdk.api_lib.storage import retry_util
from googlecloudsdk.calliope import exceptions as calliope_errors
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
import oauth2client
def _call_launch():
    return launch(apitools_download, start_byte=progress_state['start_byte'], end_byte=end_byte, additional_headers=additional_headers)