from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.pubsub import util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
from six.moves.urllib.parse import urlparse
def AddTopicDefaultsWithoutReservation(resource_ref, args, request):
    """Adds the default values for topic throughput fields with no reservation."""
    del resource_ref, args
    topic = request.topic
    if not _HasReservation(topic):
        if topic.partitionConfig is None:
            topic.partitionConfig = {}
        if topic.partitionConfig.capacity is None:
            topic.partitionConfig.capacity = {}
        capacity = topic.partitionConfig.capacity
        if capacity.publishMibPerSec is None:
            capacity.publishMibPerSec = 4
        if capacity.subscribeMibPerSec is None:
            capacity.subscribeMibPerSec = 8
    return request