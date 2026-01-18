from __future__ import absolute_import, division, print_function
import re
import socket
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
def wait_for_peer(host):
    for x in range(0, 4):
        peers = get_peers()
        if host in peers and peers[host][1].lower().find('peer in cluster') != -1:
            return True
        time.sleep(1)
    return False