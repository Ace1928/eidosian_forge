import platform
import re
import sys
import typing
def _linux_version_to_tuple(s: str) -> typing.Tuple[int, int, int]:
    return tuple(map(_versionatom, s.split('.')[:3]))