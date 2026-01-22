from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.util import api_enablement
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.api_lib.util import apis_util
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.generated_clients.apis import apis_map
import six
def _CheckResponse(response):
    """Checks API error.

    If it's an enablement error, prompt to enable & retry.
    If it's a resource exhausted error, no retry & return.

    Args:
      response: response that had an error.

    Raises:
      apitools_exceptions.RequestError: error which should signal apitools to
        retry.
      api_exceptions.HttpException: the parsed error.
    """
    if response is None:
        raise apitools_exceptions.RequestError('Request to url %s did not return a response.' % response.request_url)
    elif response.status_code == RESOURCE_EXHAUSTED_STATUS_CODE:
        return
    elif response.status_code >= 500:
        raise apitools_exceptions.BadStatusCodeError.FromResponse(response)
    elif response.retry_after:
        raise apitools_exceptions.RetryAfterError.FromResponse(response)
    response_as_error = apitools_exceptions.HttpError.FromResponse(response)
    if properties.VALUES.core.should_prompt_to_enable_api.GetBool():
        _CheckForApiEnablementError(response_as_error)