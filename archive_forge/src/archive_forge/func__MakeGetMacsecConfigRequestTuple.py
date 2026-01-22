from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def _MakeGetMacsecConfigRequestTuple(self):
    return (self._client.interconnects, 'GetMacsecConfig', self._messages.ComputeInterconnectsGetMacsecConfigRequest(project=self.ref.project, interconnect=self.ref.Name()))