from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.api_lib.orgpolicy import utils
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.generated_clients.apis.orgpolicy.v2 import orgpolicy_v2_messages
def ListConstraints(self, parent):
    if parent.startswith('organizations/'):
        request = self.messages.OrgpolicyOrganizationsConstraintsListRequest(parent=parent)
        return self.client.organizations_constraints.List(request)
    elif parent.startswith('folders/'):
        request = self.messages.OrgpolicyFoldersConstraintsListRequest(parent=parent)
        return self.client.folders_constraints.List(request)
    else:
        request = self.messages.OrgpolicyProjectsConstraintsListRequest(parent=parent)
        return self.client.projects_constraints.List(request)