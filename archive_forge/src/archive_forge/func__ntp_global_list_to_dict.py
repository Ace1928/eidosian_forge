from __future__ import absolute_import, division, print_function
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.resource_module import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.arista.eos.plugins.module_utils.network.eos.facts.facts import Facts
from ansible_collections.arista.eos.plugins.module_utils.network.eos.rm_templates.ntp_global import (
from ansible_collections.arista.eos.plugins.module_utils.network.eos.utils.utils import (
def _ntp_global_list_to_dict(self, entry):
    if 'authentication_keys' in entry:
        key_dict = {}
        for el in entry['authentication_keys']:
            key_dict.update({el['id']: el})
        entry['authentication_keys'] = key_dict
    if 'servers' in entry:
        server_dict = {}
        for el in entry['servers']:
            server_dict.update({el['server']: el})
        entry['servers'] = server_dict
    if 'serve' in entry:
        serve_dict = {}
        main_dict = {}
        if entry['serve'].get('all'):
            main_dict.update({'all': entry['serve']['all']})
        if entry['serve'].get('access_lists'):
            for el in entry['serve'].get('access_lists'):
                if 'acls' in el:
                    acl_dict = {}
                    for acl in el['acls']:
                        acl_dict.update({acl['acl_name']: acl})
                    el['acls'] = acl_dict
                serve_dict.update({el['afi']: el})
            if serve_dict:
                main_dict.update({'access_lists': serve_dict})
        if serve_dict:
            entry['serve'] = main_dict