from __future__ import (absolute_import, division, print_function)
import os
import time
import traceback
from ansible.module_utils._text import to_text
import json
from ansible_collections.fortinet.fortios.plugins.module_utils.common.type_utils import underscore_to_hyphen
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.secret_field import is_secret_field
def __check_version(revisions, version):
    result = dict()
    resolved_versions = list(revisions.keys())
    resolved_versions.sort(key=lambda x: int(x.split('.')[0][1]) * 10000 + int(x.split('.')[1]) * 100 + int(x.split('.')[2]))
    nearest_index = -1
    for i in range(len(resolved_versions)):
        if resolved_versions[i] <= version:
            nearest_index = i
    if nearest_index == -1:
        result['supported'] = False
        result['reason'] = 'not supported until in %s' % resolved_versions[0]
    elif revisions[resolved_versions[nearest_index]] is False:
        latest_index = -1
        for i in range(nearest_index + 1, len(resolved_versions)):
            if revisions[resolved_versions[i]] is True:
                latest_index = i
                break
        earliest_index = nearest_index
        while earliest_index >= 0:
            if revisions[resolved_versions[earliest_index]] is True:
                break
            earliest_index -= 1
        earliest_index = 0 if earliest_index < 0 else earliest_index
        if latest_index == -1:
            result['reason'] = 'not supported since %s' % resolved_versions[earliest_index]
        else:
            result['reason'] = 'not supported since %s, before %s' % (resolved_versions[earliest_index], resolved_versions[latest_index])
        result['supported'] = False
    else:
        result['supported'] = True
    return result