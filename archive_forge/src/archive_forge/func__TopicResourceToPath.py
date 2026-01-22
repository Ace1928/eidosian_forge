from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from google.cloud.pubsublite import cloudpubsub
from google.cloud.pubsublite import types
from google.cloud.pubsublite.cloudpubsub import message_transforms
from googlecloudsdk.api_lib.pubsub import topics
from googlecloudsdk.command_lib.pubsub import lite_util
from googlecloudsdk.core import gapic_util
from googlecloudsdk.core.util import http_encoding
def _TopicResourceToPath(self, resource):
    return types.TopicPath(project=lite_util.ProjectIdToProjectNumber(resource.projectsId), location=lite_util.LocationToZoneOrRegion(resource.locationsId), name=resource.topicsId)