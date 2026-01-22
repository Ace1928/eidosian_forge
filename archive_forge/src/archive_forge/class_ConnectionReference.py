import collections
import re
import sys
from typing import Any, Optional, Tuple, Type, Union
from absl import app
from pyglib import stringutil
class ConnectionReference(Reference):
    _required_fields = frozenset(('projectId', 'location', 'connectionId'))
    _format_str = '%(projectId)s.%(location)s.%(connectionId)s'
    _path_str = 'projects/%(projectId)s/locations/%(location)s/connections/%(connectionId)s'
    typename = 'connection'

    def path(self) -> str:
        return self._path_str % dict(self)