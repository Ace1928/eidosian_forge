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
def _CheckForApiEnablementError(response_as_error):
    enablement_info = GetApiEnablementInfo(response_as_error)
    if enablement_info:
        if state['already_prompted_to_enable'] or skip_activation_prompt:
            raise apitools_exceptions.RequestError('Retry')
        state['already_prompted_to_enable'] = True
        PromptToEnableApi(*enablement_info)