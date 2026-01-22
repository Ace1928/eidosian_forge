from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.notebooks import util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def GetEnvironmentURI(resource):
    environment = resources.REGISTRY.ParseRelativeName(resource.name, collection='notebooks.projects.locations.environments')
    return environment.SelfLink()