from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.hwc_utils import (
def adjust_data_volumes(parent_input, parent_cur):
    iv = parent_input.get('data_volumes')
    if not (iv and isinstance(iv, list)):
        return
    cv = parent_cur.get('data_volumes')
    if not (cv and isinstance(cv, list)):
        return
    lcv = len(cv)
    result = []
    q = []
    for iiv in iv:
        if len(q) == lcv:
            break
        icv = None
        for j in range(lcv):
            if j in q:
                continue
            icv = cv[j]
            if iiv['volume_id'] != icv['volume_id']:
                continue
            result.append(icv)
            q.append(j)
            break
        else:
            break
    if len(q) != lcv:
        for i in range(lcv):
            if i not in q:
                result.append(cv[i])
    if len(result) != lcv:
        raise Exception('adjust property(data_volumes) failed, the array number is not equal')
    parent_cur['data_volumes'] = result