from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.py import exceptions
from googlecloudsdk.api_lib.compute import operation_quota_utils
from googlecloudsdk.api_lib.compute import utils
import six
def _GenerateErrorMessage(exception):
    """Generate Error Message given exception."""
    error_message = None
    try:
        data = json.loads(exception.content)
        if isinstance(exception, exceptions.HttpError) and utils.JsonErrorHasDetails(data):
            error_message = (exception.status_code, BuildMessageForErrorWithDetails(data))
        else:
            error_message = (exception.status_code, data.get('error', {}).get('message'))
    except ValueError:
        pass
    if not error_message:
        error_message = (exception.status_code, exception.content)
    return error_message