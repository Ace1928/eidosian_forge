from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.core.exceptions import Error
import six
def GetInstantSnapshotResource(self):
    request_msg = self.GetInstantSnapshotRequestMessage()
    return self._service.Get(request_msg)