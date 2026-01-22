from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.eventarc import common
from googlecloudsdk.api_lib.runtime_config import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def GetLocationsURI(resource):
    location = resources.REGISTRY.ParseRelativeName(resource.name, collection='eventarc.projects.locations')
    return location.SelfLink()