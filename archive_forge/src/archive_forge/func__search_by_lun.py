from __future__ import (absolute_import, division, print_function)
import json
import os
import subprocess
import time
import traceback
import inspect
from ansible.module_utils.six.moves.http_client import HTTPSConnection, IncompleteRead
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def _search_by_lun(disks_service, lun_id):
    """
    Find disk by LUN ID.
    """
    res = [disk for disk in disks_service.list(search='disk_type=lun') if disk.lun_storage.id == lun_id]
    return res[0] if res else None