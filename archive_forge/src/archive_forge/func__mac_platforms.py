import re
from typing import List, Optional, Tuple
from pip._vendor.packaging.tags import (
def _mac_platforms(arch: str) -> List[str]:
    match = _osx_arch_pat.match(arch)
    if match:
        name, major, minor, actual_arch = match.groups()
        mac_version = (int(major), int(minor))
        arches = ['{}_{}'.format(name, arch[len('macosx_'):]) for arch in mac_platforms(mac_version, actual_arch)]
    else:
        arches = [arch]
    return arches