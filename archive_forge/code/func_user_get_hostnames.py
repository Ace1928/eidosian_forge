from __future__ import (absolute_import, division, print_function)
import string
import json
import re
from ansible.module_utils.six import iteritems
from ansible_collections.community.mysql.plugins.module_utils.mysql import (
def user_get_hostnames(cursor, user):
    cursor.execute('SELECT Host FROM mysql.user WHERE user = %s', (user,))
    hostnames_raw = cursor.fetchall()
    hostnames = []
    for hostname_raw in hostnames_raw:
        hostnames.append(hostname_raw[0])
    return hostnames