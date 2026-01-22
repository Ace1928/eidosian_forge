from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.builds import flags as build_flags
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _CreateOrUpdateTrigger(self, client, messages, project, location, trigger):
    if trigger.id:
        return self._UpdateTrigger(client, messages, project, location, trigger)
    elif trigger.name:
        try:
            return self._UpdateTrigger(client, messages, project, location, trigger)
        except apitools_exceptions.HttpNotFoundError:
            return self._CreateTrigger(client, messages, project, location, trigger)
    else:
        return self._CreateTrigger(client, messages, project, location, trigger)