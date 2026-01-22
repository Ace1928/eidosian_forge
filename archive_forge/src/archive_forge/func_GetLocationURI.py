from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.notebooks import util
def GetLocationURI(resource):
    location_resource = util.GetLocationResource(resource.name)
    return location_resource.SelfLink()