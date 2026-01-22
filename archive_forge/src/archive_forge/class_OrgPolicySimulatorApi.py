from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
class OrgPolicySimulatorApi(object):
    """Base Class for OrgPolicy Simulator API."""

    def __new__(cls, release_track):
        if release_track == base.ReleaseTrack.ALPHA:
            return super(OrgPolicySimulatorApi, cls).__new__(OrgPolicySimulatorApiAlpha)
        if release_track == base.ReleaseTrack.BETA:
            return super(OrgPolicySimulatorApi, cls).__new__(OrgPolicySimulatorApiBeta)
        if release_track == base.ReleaseTrack.GA:
            return super(OrgPolicySimulatorApi, cls).__new__(OrgPolicySimulatorApiGA)

    def __init__(self, release_track):
        self.api_version = GetApiVersion(release_track)
        self.client = apis.GetClientInstance(_API_NAME, self.api_version)
        self.messages = apis.GetMessagesModule(_API_NAME, self.api_version)

    def GetViolationsPreviewId(self, operation_name):
        return operation_name.split('/')[-3]

    def WaitForOperation(self, operation, message):
        """Wait for the operation to complete."""
        v1_client = apis.GetClientInstance(_API_NAME, self.api_version)
        registry = resources.REGISTRY.Clone()
        registry.RegisterApiByName('policysimulator', self.api_version)
        operation_ref = registry.Parse(operation.name, params={'organizationsId': properties.VALUES.access_context_manager.organization.GetOrFail, 'locationsId': 'global', 'orgPolicyViolationsPreviewsId': self.GetViolationsPreviewId(operation.name)}, collection='policysimulator.organizations.locations.orgPolicyViolationsPreviews.operations')
        poller = waiter.CloudOperationPollerNoResources(v1_client.operations)
        return waiter.WaitFor(poller, operation_ref, message, wait_ceiling_ms=_MAX_WAIT_TIME_MS)

    @abc.abstractmethod
    def CreateOrgPolicyViolationsPreviewRequest(self, violations_preview=None, parent=None):
        pass

    @abc.abstractmethod
    def GetPolicysimulatorOrgPolicyViolationsPreview(self, name=None, overlay=None, resource_counts=None, state=None, violations_count=None):
        pass

    @abc.abstractmethod
    def GetOrgPolicyOverlay(self, custom_constraints=None, policies=None):
        pass

    @abc.abstractmethod
    def GetOrgPolicyPolicyOverlay(self, policy=None, policy_parent=None):
        pass

    @abc.abstractmethod
    def GetOrgPolicyCustomConstraintOverlay(self, custom_constraint=None, custom_constraint_parent=None):
        pass

    @abc.abstractmethod
    def GetOrgPolicyViolationsPreviewMessage(self):
        pass