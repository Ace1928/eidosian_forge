from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.builds import flags as build_flags
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _CreateTrigger(self, client, messages, project, location, trigger):
    parent = resources.REGISTRY.Create(collection='cloudbuild.projects.locations', projectsId=project, locationsId=location).RelativeName()
    return client.projects_locations_triggers.Create(messages.CloudbuildProjectsLocationsTriggersCreateRequest(parent=parent, buildTrigger=trigger))