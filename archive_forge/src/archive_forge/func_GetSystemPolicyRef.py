from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def GetSystemPolicyRef(location):
    return resources.REGISTRY.Parse(None, params={'locationsId': location}, collection=LOCATIONS_POLICY)