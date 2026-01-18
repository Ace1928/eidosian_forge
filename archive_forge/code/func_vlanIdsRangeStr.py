from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def vlanIdsRangeStr(self, vlanList):
    rangeList = []
    for vid in vlanList:
        if '-' in vid:
            vidList = vid.split('-')
            lower = int(vidList[0])
            upper = int(vidList[1])
            for i in range(lower, upper + 1):
                rangeList.append(str(i))
        else:
            rangeList.append(vid)
    return rangeList