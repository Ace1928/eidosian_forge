import collections
import re
import sys
from typing import Any, Optional, Tuple, Type, Union
from absl import app
from pyglib import stringutil
class CapacityCommitmentReference(Reference):
    """Helper class to provide a reference to capacity commitment."""
    _required_fields = frozenset(('projectId', 'location', 'capacityCommitmentId'))
    _format_str = '%(projectId)s:%(location)s.%(capacityCommitmentId)s'
    _path_str = 'projects/%(projectId)s/locations/%(location)s/capacityCommitments/%(capacityCommitmentId)s'
    typename = 'capacity commitment'

    def path(self) -> str:
        return self._path_str % dict(self)