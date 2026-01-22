from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import resources
def LiensRegistry():
    registry = resources.REGISTRY.Clone()
    registry.RegisterApiByName('cloudresourcemanager', LIENS_API_VERSION)
    return registry