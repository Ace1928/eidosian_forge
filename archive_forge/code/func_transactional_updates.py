from __future__ import absolute_import, division, print_function
import os.path
import xml
import re
from xml.dom.minidom import parseString as parseXML
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import AnsibleModule
def transactional_updates():
    return os.path.exists('/usr/sbin/transactional-update') and get_fs_type_and_readonly_state('/') == ('btrfs', True)