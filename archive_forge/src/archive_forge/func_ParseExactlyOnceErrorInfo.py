from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.pubsub import subscriptions
from googlecloudsdk.api_lib.pubsub import topics
from googlecloudsdk.api_lib.util import exceptions as exc
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.resource import resource_projector
from googlecloudsdk.core.util import times
import six
def ParseExactlyOnceErrorInfo(error_metadata):
    """Parses error metadata for exactly once ack/modAck failures.

  Args:
    error_metadata: error metadata as dict of format ack_id -> failure_reason.

  Returns:
    list: error metadata with only exactly once failures.
  """
    ack_ids_and_failure_reasons = []
    for error_md in error_metadata:
        if 'reason' not in error_md or 'EXACTLY_ONCE' not in error_md['reason']:
            continue
        if 'metadata' not in error_md or not isinstance(error_md['metadata'], dict):
            continue
        for ack_id, failure_reason in error_md['metadata'].items():
            if 'PERMANENT_FAILURE' in failure_reason or 'TEMPORARY_FAILURE' in failure_reason:
                result = resource_projector.MakeSerializable({})
                result['AckId'] = ack_id
                result['FailureReason'] = failure_reason
                ack_ids_and_failure_reasons.append(result)
    return ack_ids_and_failure_reasons