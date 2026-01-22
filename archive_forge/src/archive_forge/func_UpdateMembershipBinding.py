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
def UpdateMembershipBinding(self, name, scope, labels, mask):
    """Updates a Membership-Binding resource.

    Args:
      name: the Binding name.
      scope: the Scope associated with binding.
      labels: the labels for the Membership Binding resource.
      mask: a mask of the fields to update.

    Returns:
      An operation

    Raises:
      apitools.base.py.HttpError: if the request returns an HTTP error
    """
    binding = self.messages.MembershipBinding(name=name, scope=scope, labels=labels)
    req = self.messages.GkehubProjectsLocationsMembershipsBindingsPatchRequest(membershipBinding=binding, name=binding.name, updateMask=mask)
    op = self.client.projects_locations_memberships_bindings.Patch(req)
    op_resource = resources.REGISTRY.ParseRelativeName(op.name, collection='gkehub.projects.locations.operations')
    return waiter.WaitFor(waiter.CloudOperationPoller(self.client.projects_locations_memberships_bindings, self.client.projects_locations_operations), op_resource, 'Waiting for membership binding to be updated')