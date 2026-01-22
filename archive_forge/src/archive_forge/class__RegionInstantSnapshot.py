from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.core.exceptions import Error
import six
class _RegionInstantSnapshot(_CommonInstantSnapshot):
    """A wrapper for Compute Engine RegionInstantSnapshotService API client."""

    def __init__(self, client, ips_ref, messages):
        _CommonInstantSnapshot.__init__(self)
        self._ips_ref = ips_ref
        self._client = client
        self._service = client.regionInstantSnapshots
        self._messages = messages

    @classmethod
    def GetOperationCollection(cls):
        return 'compute.regionOperations'

    def GetInstantSnapshotRequestMessage(self):
        return self._messages.ComputeRegionInstantSnapshotsGetRequest(**self._ips_ref.AsDict())

    def GetSetLabelsRequestMessage(self):
        return self._messages.RegionSetLabelsRequest

    def GetSetInstantSnapshotLabelsRequestMessage(self, ips, labels):
        req = self._messages.ComputeRegionInstantSnapshotsSetLabelsRequest
        return req(project=self._ips_ref.project, resource=self._ips_ref.instantSnapshot, region=self._ips_ref.region, regionSetLabelsRequest=self._messages.RegionSetLabelsRequest(labelFingerprint=ips.labelFingerprint, labels=labels))