from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.app import util
from googlecloudsdk.api_lib.tasks import task_queues_convertors as convertors
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.tasks import app
from googlecloudsdk.command_lib.tasks import constants
from googlecloudsdk.command_lib.tasks import flags
from googlecloudsdk.command_lib.tasks import parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
import six
from six.moves import urllib
def _DoesAttributeNeedToBeUpdated(cur_queue_state, attribute, new_value):
    """Checks whether the attribute & value provided need to be updated.

  Note: We only check if the attribute exists in `queue.rateLimits` and
  `queue.retryConfig` since those are the only attributes we verify here. The
  only attribute we do not verify here is app-engine routing override which we
  handle separately.

  Args:
    cur_queue_state: apis.cloudtasks.<ver>.cloudtasks_<ver>_messages.Queue,
      The Queue instance fetched from the backend.
    attribute: Snake case representation of the CT API attribute name. One
      example is 'max_burst_size'.
    new_value: The value we are trying to set this attribute to.

  Returns:
    True if the attribute needs to be updated to the new value, False otherwise.
  """
    proto_attribute_name = convertors.ConvertStringToCamelCase(attribute)
    if hasattr(cur_queue_state, 'rateLimits') and hasattr(cur_queue_state.rateLimits, proto_attribute_name):
        old_value = getattr(cur_queue_state.rateLimits, proto_attribute_name)
    elif hasattr(cur_queue_state.retryConfig, proto_attribute_name):
        old_value = getattr(cur_queue_state.retryConfig, proto_attribute_name)
    else:
        return True
    if old_value == new_value:
        return False
    if old_value is None and attribute != 'max_concurrent_dispatches' and (attribute in constants.PUSH_QUEUES_APP_DEPLOY_DEFAULT_VALUES) and (new_value == constants.PUSH_QUEUES_APP_DEPLOY_DEFAULT_VALUES[attribute]):
        return False
    if attribute == 'max_dispatches_per_second' and (not new_value):
        return False
    if old_value is None or new_value is None:
        return True
    old_value = convertors.CheckAndConvertStringToFloatIfApplicable(old_value)
    new_value = convertors.CheckAndConvertStringToFloatIfApplicable(new_value)
    if isinstance(old_value, float) and isinstance(new_value, float):
        return not IsClose(old_value, new_value)
    return old_value != new_value