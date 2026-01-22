from __future__ import absolute_import, division, print_function
import os
import plistlib
from abc import ABCMeta, abstractmethod
from time import sleep
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
@staticmethod
def __find_service_plist(service_name):
    """Finds the plist file associated with a service"""
    launchd_paths = [os.path.join(os.getenv('HOME'), 'Library/LaunchAgents'), '/Library/LaunchAgents', '/Library/LaunchDaemons', '/System/Library/LaunchAgents', '/System/Library/LaunchDaemons']
    for path in launchd_paths:
        try:
            files = os.listdir(path)
        except OSError:
            continue
        filename = '%s.plist' % service_name
        if filename in files:
            return os.path.join(path, filename)
    return None