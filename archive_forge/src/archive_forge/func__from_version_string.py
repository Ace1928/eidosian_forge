from functools import total_ordering
from ._funcs import astuple
from ._make import attrib, attrs
@classmethod
def _from_version_string(cls, s):
    """
        Parse *s* and return a _VersionInfo.
        """
    v = s.split('.')
    if len(v) == 3:
        v.append('final')
    return cls(year=int(v[0]), minor=int(v[1]), micro=int(v[2]), releaselevel=v[3])