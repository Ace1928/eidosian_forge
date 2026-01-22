from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import sys
import uuid
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import extra_types
from googlecloudsdk.api_lib.infra_manager import configmanager_util
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.infra_manager import deterministic_snapshot
from googlecloudsdk.command_lib.infra_manager import errors
from googlecloudsdk.command_lib.infra_manager import staging_bucket_util
from googlecloudsdk.core import log
from googlecloudsdk.core import requests
from googlecloudsdk.core import resources
from googlecloudsdk.core.resource import resource_transform
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def ExportPreviewResult(messages, preview_full_name, file=None):
    """Creates a signed uri to download the preview results.

  Args:
    messages: ModuleType, the messages module that lets us form Infra Manager
      API messages based on our protos.
    preview_full_name: string, the fully qualified name of the preview,
      e.g. "projects/p/locations/l/previews/p".]
    file: string, the file name to download results to.
  Returns:
    A messages.ExportPreviewResultResponse which contains signed uri to
    download binary and json plan files.
  """
    export_preview_result_request = messages.ExportPreviewResultRequest()
    log.status.Print('Initiating export preview results...')
    preview_result_resp = configmanager_util.ExportPreviewResult(export_preview_result_request, preview_full_name)
    if file is None:
        return preview_result_resp
    binary_data = requests.GetSession().get(preview_result_resp.result.binarySignedUri)
    json_data = requests.GetSession().get(preview_result_resp.result.jsonSignedUri)
    if file.endswith(os.sep):
        file += 'preview'
    files.WriteBinaryFileContents(file + '.tfplan', binary_data.content)
    files.WriteBinaryFileContents(file + '.json', json_data.content)
    log.status.Print(f'Exported preview artifacts {file}.tfplan and {file}.json')
    return