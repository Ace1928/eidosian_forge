from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def _MakeGetDiagnosticsRequestTuple(self):
    return (self._client.interconnects, 'GetDiagnostics', self._messages.ComputeInterconnectsGetDiagnosticsRequest(project=self.ref.project, interconnect=self.ref.Name()))