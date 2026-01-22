from __future__ import (absolute_import, division, print_function)
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts.collector import BaseFactCollector
class SelinuxFactCollector(BaseFactCollector):
    name = 'selinux'
    _fact_ids = set()

    def collect(self, module=None, collected_facts=None):
        facts_dict = {}
        selinux_facts = {}
        if not HAVE_SELINUX:
            selinux_facts['status'] = 'Missing selinux Python library'
            facts_dict['selinux'] = selinux_facts
            facts_dict['selinux_python_present'] = False
            return facts_dict
        facts_dict['selinux_python_present'] = True
        if not selinux.is_selinux_enabled():
            selinux_facts['status'] = 'disabled'
        else:
            selinux_facts['status'] = 'enabled'
            try:
                selinux_facts['policyvers'] = selinux.security_policyvers()
            except (AttributeError, OSError):
                selinux_facts['policyvers'] = 'unknown'
            try:
                rc, configmode = selinux.selinux_getenforcemode()
                if rc == 0:
                    selinux_facts['config_mode'] = SELINUX_MODE_DICT.get(configmode, 'unknown')
                else:
                    selinux_facts['config_mode'] = 'unknown'
            except (AttributeError, OSError):
                selinux_facts['config_mode'] = 'unknown'
            try:
                mode = selinux.security_getenforce()
                selinux_facts['mode'] = SELINUX_MODE_DICT.get(mode, 'unknown')
            except (AttributeError, OSError):
                selinux_facts['mode'] = 'unknown'
            try:
                rc, policytype = selinux.selinux_getpolicytype()
                if rc == 0:
                    selinux_facts['type'] = policytype
                else:
                    selinux_facts['type'] = 'unknown'
            except (AttributeError, OSError):
                selinux_facts['type'] = 'unknown'
        facts_dict['selinux'] = selinux_facts
        return facts_dict