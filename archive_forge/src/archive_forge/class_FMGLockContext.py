from __future__ import (absolute_import, division, print_function)
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import FMGR_RC
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import FMGBaseException
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import FMGRCommon
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import scrub_dict
class FMGLockContext(object):
    """
    - DEPRECATING: USING CONNECTION MANAGER NOW INSTEAD. EVENTUALLY THIS CLASS WILL DISAPPEAR. PLEASE
    - CONVERT ALL MODULES TO CONNECTION MANAGER METHOD.
    - LEGACY pyFMG HANDLER OBJECT: REQUIRES A CHECK FOR PY FMG AT TOP OF PAGE
    """

    def __init__(self, fmg):
        self._fmg = fmg
        self._locked_adom_list = list()
        self._uses_workspace = False
        self._uses_adoms = False

    @property
    def uses_workspace(self):
        return self._uses_workspace

    @uses_workspace.setter
    def uses_workspace(self, val):
        self._uses_workspace = val

    @property
    def uses_adoms(self):
        return self._uses_adoms

    @uses_adoms.setter
    def uses_adoms(self, val):
        self._uses_adoms = val

    def add_adom_to_lock_list(self, adom):
        if adom not in self._locked_adom_list:
            self._locked_adom_list.append(adom)

    def remove_adom_from_lock_list(self, adom):
        if adom in self._locked_adom_list:
            self._locked_adom_list.remove(adom)

    def check_mode(self):
        url = '/cli/global/system/global'
        code, resp_obj = self._fmg.get(url, fields=['workspace-mode', 'adom-status'])
        try:
            if resp_obj['workspace-mode'] != 0:
                self.uses_workspace = True
        except KeyError:
            self.uses_workspace = False
        try:
            if resp_obj['adom-status'] == 1:
                self.uses_adoms = True
        except KeyError:
            self.uses_adoms = False

    def run_unlock(self):
        for adom_locked in self._locked_adom_list:
            self.unlock_adom(adom_locked)

    def lock_adom(self, adom=None, *args, **kwargs):
        if adom:
            if adom.lower() == 'global':
                url = '/dvmdb/global/workspace/lock/'
            else:
                url = '/dvmdb/adom/{adom}/workspace/lock/'.format(adom=adom)
        else:
            url = '/dvmdb/adom/root/workspace/lock'
        code, respobj = self._fmg.execute(url, {}, *args, **kwargs)
        if code == 0 and respobj['status']['message'].lower() == 'ok':
            self.add_adom_to_lock_list(adom)
        return (code, respobj)

    def unlock_adom(self, adom=None, *args, **kwargs):
        if adom:
            if adom.lower() == 'global':
                url = '/dvmdb/global/workspace/unlock/'
            else:
                url = '/dvmdb/adom/{adom}/workspace/unlock/'.format(adom=adom)
        else:
            url = '/dvmdb/adom/root/workspace/unlock'
        code, respobj = self._fmg.execute(url, {}, *args, **kwargs)
        if code == 0 and respobj['status']['message'].lower() == 'ok':
            self.remove_adom_from_lock_list(adom)
        return (code, respobj)

    def commit_changes(self, adom=None, aux=False, *args, **kwargs):
        if adom:
            if aux:
                url = '/pm/config/adom/{adom}/workspace/commit'.format(adom=adom)
            elif adom.lower() == 'global':
                url = '/dvmdb/global/workspace/commit/'
            else:
                url = '/dvmdb/adom/{adom}/workspace/commit'.format(adom=adom)
        else:
            url = '/dvmdb/adom/root/workspace/commit'
        return self._fmg.execute(url, {}, *args, **kwargs)