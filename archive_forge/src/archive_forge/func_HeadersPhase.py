import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
def HeadersPhase(self):
    headers_phase = self.GetBuildPhaseByType(PBXHeadersBuildPhase)
    if headers_phase is None:
        headers_phase = PBXHeadersBuildPhase()
        insert_at = len(self._properties['buildPhases'])
        for index, phase in enumerate(self._properties['buildPhases']):
            if isinstance(phase, PBXResourcesBuildPhase) or isinstance(phase, PBXSourcesBuildPhase) or isinstance(phase, PBXFrameworksBuildPhase):
                insert_at = index
                break
        self._properties['buildPhases'].insert(insert_at, headers_phase)
        headers_phase.parent = self
    return headers_phase