from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def update_multi_dns(module, array):
    """Update a DNS configuration"""
    changed = False
    current_dns = list(array.get_dns(names=[module.params['name']]).items)[0]
    new_dns = current_dns
    if module.params['domain'] and current_dns.domain != module.params['domain']:
        new_dns.domain = module.params['domain']
        changed = True
    if module.params['service'] and current_dns.services != [module.params['service']]:
        module.fail_json(msg='Changing service type is not permitted')
    if module.params['nameservers'] and sorted(current_dns.nameservers) != sorted(module.params['nameservers']):
        new_dns.nameservers = module.params['nameservers']
        changed = True
    if (module.params['source'] or module.params['source'] == '') and current_dns.source.name != module.params['source']:
        new_dns.source.name = module.params['source']
        changed = True
    if changed and (not module.check_mode):
        res = array.patch_dns(names=[module.params['name']], dns=flasharray.Dns(domain=new_dns.domain, nameservers=new_dns.nameservers, source=flasharray.ReferenceNoId(module.params['source'])))
        if res.status_code != 200:
            module.fail_json(msg='Update to DNS service {0} failed. Error: {1}'.format(module.params['name'], res.errors[0].message))
    module.exit_json(changed=changed)