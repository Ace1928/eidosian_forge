from __future__ import (absolute_import, division, print_function)
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import FMGR_RC
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import FMGBaseException
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import FMGRCommon
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import scrub_dict
class AnsibleFortiManager(object):
    """
    - DEPRECATING: USING CONNECTION MANAGER NOW INSTEAD. EVENTUALLY THIS CLASS WILL DISAPPEAR. PLEASE
    - CONVERT ALL MODULES TO CONNECTION MANAGER METHOD.
    - LEGACY pyFMG HANDLER OBJECT: REQUIRES A CHECK FOR PY FMG AT TOP OF PAGE
    """

    def __init__(self, module, ip=None, username=None, passwd=None, use_ssl=True, verify_ssl=False, timeout=300):
        self.ip = ip
        self.username = username
        self.passwd = passwd
        self.use_ssl = use_ssl
        self.verify_ssl = verify_ssl
        self.timeout = timeout
        self.fmgr_instance = None
        if not HAS_PYFMGR:
            module.fail_json(msg='Could not import the python library pyFMG required by this module')
        self.module = module

    def login(self):
        if self.ip is not None:
            self.fmgr_instance = FortiManager(self.ip, self.username, self.passwd, use_ssl=self.use_ssl, verify_ssl=self.verify_ssl, timeout=self.timeout, debug=False, disable_request_warnings=True)
            return self.fmgr_instance.login()

    def logout(self):
        if self.fmgr_instance.sid is not None:
            self.fmgr_instance.logout()

    def get(self, url, data):
        return self.fmgr_instance.get(url, **data)

    def set(self, url, data):
        return self.fmgr_instance.set(url, **data)

    def update(self, url, data):
        return self.fmgr_instance.update(url, **data)

    def delete(self, url, data):
        return self.fmgr_instance.delete(url, **data)

    def add(self, url, data):
        return self.fmgr_instance.add(url, **data)

    def execute(self, url, data):
        return self.fmgr_instance.execute(url, **data)

    def move(self, url, data):
        return self.fmgr_instance.move(url, **data)

    def clone(self, url, data):
        return self.fmgr_instance.clone(url, **data)