from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from typing import Generator
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
from googlecloudsdk.generated_clients.apis.gkehub.v1alpha import gkehub_v1alpha_messages as messages
def CreateFeature(self, parent, feature_id, feature):
    """Creates a Feature and returns the long-running operation message.

    Args:
      parent: The parent in the form /projects/*/locations/*.
      feature_id: The short ID for this Feature in the Hub API.
      feature: A Feature message specifying the Feature data to create.

    Returns:
      The long running operation reference. Use the feature_waiter and
      OperationRef to watch the operation and get the final status, typically
      using waiter.WaitFor to present a user-friendly spinner.
    """
    req = self.messages.GkehubProjectsLocationsFeaturesCreateRequest(feature=feature, featureId=feature_id, parent=parent)
    return self.client.projects_locations_features.Create(req)