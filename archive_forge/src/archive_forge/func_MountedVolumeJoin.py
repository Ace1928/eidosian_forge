from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from typing import Mapping, Sequence
from googlecloudsdk.api_lib.run import k8s_object
def MountedVolumeJoin(self, subgroup=None):
    vols = self._volumes
    mounts = self.volume_mounts
    if subgroup:
        vols = getattr(vols, subgroup)
        mounts = getattr(mounts, subgroup)
    return {path: vols.get(vol) for path, vol in mounts.items()}