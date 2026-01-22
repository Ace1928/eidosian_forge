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
def ParseExactlyOnceAckIdsAndFailureReasons(ack_ids_and_failure_reasons, ack_ids):
    failed_ack_ids = [ack['AckId'] for ack in ack_ids_and_failure_reasons]
    successfully_processed_ack_ids = [ack_id for ack_id in ack_ids if ack_id not in failed_ack_ids]
    return (failed_ack_ids, successfully_processed_ack_ids)