from __future__ import (absolute_import, division, print_function)
import bisect
import json
import pkgutil
import re
from ansible import constants as C
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.distro import LinuxDistribution
from ansible.utils.display import Display
from ansible.utils.plugin_docs import get_versioned_doclink
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.facts.system.distribution import Distribution
from traceback import format_exc
def _version_fuzzy_match(version, version_map):
    res = version_map.get(version)
    if res:
        return res
    sorted_looseversions = sorted([LooseVersion(v) for v in version_map.keys()])
    find_looseversion = LooseVersion(version)
    kpos = bisect.bisect(sorted_looseversions, find_looseversion)
    if kpos == 0:
        return version_map.get(sorted_looseversions[0].vstring)
    return version_map.get(sorted_looseversions[kpos - 1].vstring)