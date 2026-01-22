from __future__ import absolute_import, division, print_function
from collections import OrderedDict
import base64
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_text, to_bytes, to_native
from ansible.module_utils.six.moves.urllib.error import HTTPError, URLError
from ansible.module_utils.urls import open_url, ConnectionError, SSLValidationError
def assemble_json(cpmmodule, existing_interface):
    total_block = total_indices = 0
    is_clear = is_changed = protocol = loop = 0
    json_load = ''
    ietfstring = 'ietf-ipv4'
    syslogenable = syslogport = syslogsecure = None
    syslogtransport = None
    user_load = ''
    indices = []
    blockarray = []
    for x in range(0, 48):
        indices.insert(x, None)
        blockarray.insert(x, None)
    ports = cpmmodule.params['interface']
    if cpmmodule.params['clear'] is not None:
        is_clear = int(cpmmodule.params['clear'])
    if cpmmodule.params['protocol'] is not None:
        protocol = int(cpmmodule.params['protocol'])
        if protocol == 1:
            ietfstring = 'ietf-ipv6'
    if cpmmodule.params['enable'] is not None:
        syslogenable = int(cpmmodule.params['enable'])
    if cpmmodule.params['port'] is not None:
        syslogport = int(cpmmodule.params['port'])
    if cpmmodule.params['transport'] is not None:
        syslogtransport = int(cpmmodule.params['transport'])
    if cpmmodule.params['secure'] is not None:
        syslogsecure = int(cpmmodule.params['secure'])
    index = cpmmodule.params['index']
    if index is not None:
        if isinstance(index, list):
            for x in index:
                indices.insert(total_indices, int(to_native(x)) - 1)
                total_indices += 1
    total_block = 0
    blockarray = cpmmodule.params['address']
    if blockarray is not None:
        if isinstance(blockarray, list):
            for x in blockarray:
                blockarray[total_block] = to_native(x)
                total_block += 1
    if total_indices > 0:
        if total_block != total_indices:
            return (is_changed, None)
    for x in range(0, total_block):
        if blockarray[x] is not None:
            if loop > 0:
                user_load = '%s,' % user_load
            user_load = '%s{"index": "%d"' % (user_load, indices[x] + 1)
            if blockarray[x] is not None:
                if existing_interface['syslogserver'][ports][0][ietfstring]['block'][indices[x]]['address'] != blockarray[x]:
                    is_changed = True
                user_load = '%s,"address": "%s"' % (user_load, blockarray[x])
            else:
                user_load = '%s,"address": "%s"' % (user_load, existing_interface['syslogserver'][ports][0][ietfstring]['block'][indices[x]]['address'])
            user_load = '%s}' % user_load
            loop += 1
    json_load = '{"syslogserver": [{"%s": { "%s": { "clear": %d, "change": %d' % (ports, ietfstring, is_clear, is_changed)
    if syslogenable is not None:
        if int(existing_interface['syslogserver'][ports][0][ietfstring]['enable']) != syslogenable:
            is_changed = True
        json_load = '%s, "enable": %d' % (json_load, syslogenable)
    else:
        json_load = '%s,"enable": "%s"' % (json_load, existing_interface['syslogserver'][ports][0][ietfstring]['enable'])
    if syslogport is not None:
        if int(existing_interface['syslogserver'][ports][0][ietfstring]['port']) != syslogport:
            is_changed = True
        json_load = '%s, "port": %d' % (json_load, syslogport)
    else:
        json_load = '%s,"port": "%s"' % (json_load, existing_interface['syslogserver'][ports][0][ietfstring]['port'])
    if syslogtransport is not None:
        if int(existing_interface['syslogserver'][ports][0][ietfstring]['transport']) != syslogtransport:
            is_changed = True
        json_load = '%s, "transport": %d' % (json_load, syslogtransport)
    else:
        json_load = '%s,"transport": "%s"' % (json_load, existing_interface['syslogserver'][ports][0][ietfstring]['transport'])
    if syslogsecure is not None:
        if int(existing_interface['syslogserver'][ports][0][ietfstring]['secure']) != syslogsecure:
            is_changed = True
        json_load = '%s, "secure": %d' % (json_load, syslogsecure)
    else:
        json_load = '%s,"secure": "%s"' % (json_load, existing_interface['syslogserver'][ports][0][ietfstring]['secure'])
    if len(user_load) > 0:
        json_load = '%s, "block": [ %s ]' % (json_load, user_load)
    json_load = '%s}}}]}' % json_load
    return (is_changed, json_load)