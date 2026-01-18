from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
def vdisk_probe(self, data):
    self.log('Entering function vdisk_probe')
    props = []
    resizevolume_flag = False
    if self.type == 'local hyperswap' and self.vdisk_type == 'standard mirror':
        self.module.fail_json(msg='You cannot update the topolgy from standard mirror to HyperSwap')
    if (self.vdisk_type == 'local hyperswap' or self.vdisk_type == 'standard mirror') and self.size:
        size_in_bytes = int(self.size) * 1024 * 1024
        existing_size = int(data[0]['capacity'])
        if size_in_bytes != existing_size:
            resizevolume_flag = True
            props += ['resizevolume']
        if size_in_bytes > existing_size:
            self.changebysize = size_in_bytes - existing_size
            self.expand_flag = True
        elif size_in_bytes < existing_size:
            self.changebysize = existing_size - size_in_bytes
            self.shrink_flag = True
    if self.poolA and self.poolB:
        if self.vdisk_type == 'local hyperswap' and self.type == 'standard':
            self.module.fail_json(msg='HyperSwap Volume cannot be converted to standard mirror')
        if self.vdisk_type == 'standard mirror' or self.vdisk_type == 'local hyperswap':
            if (self.poolA == self.discovered_poolA or self.poolA == self.discovered_poolB) and (self.poolB == self.discovered_poolA or self.poolB == self.discovered_poolB) and (not resizevolume_flag):
                return props
            elif not resizevolume_flag:
                self.module.fail_json(msg='Pools for Standard Mirror or HyperSwap volume cannot be updated')
        elif self.vdisk_type == 'standard' and self.type == 'local hyperswap':
            if self.poolA == self.discovered_standard_vol_pool or self.poolB == self.discovered_standard_vol_pool:
                props += ['addvolumecopy']
            else:
                self.module.fail_json(msg='One of the input pools must belong to the Volume')
        elif self.vdisk_type == 'standard' and self.type == 'standard':
            if self.poolA == self.discovered_standard_vol_pool or self.poolB == self.discovered_standard_vol_pool:
                props += ['addvdiskcopy']
            else:
                self.module.fail_json(msg='One of the input pools must belong to the Volume')
        elif self.vdisk_type and (not self.type):
            self.module.fail_json(msg='missing required argument: type')
    elif not self.poolA or not self.poolB:
        if self.vdisk_type == 'standard':
            if self.poolA == self.discovered_standard_vol_pool or self.poolB == self.discovered_standard_vol_pool:
                self.log('Standard Volume already exists, no modifications done')
                return props
        if self.poolA:
            if self.poolA == self.discovered_poolA or self.poolA == self.discovered_poolB:
                props += ['rmvolumecopy']
            else:
                self.module.fail_json(msg='One of the input pools must belong to the Volume')
        elif self.poolB:
            if self.poolB == self.discovered_poolA or self.poolB == self.discovered_poolB:
                props += ['rmvolumecopy']
            else:
                self.module.fail_json(msg='One of the input pools must belong to the Volume')
    if not (self.poolA or not self.poolB) and (not self.size):
        if self.system_topology == 'hyperswap' and self.type == 'local hyperswap':
            self.module.fail_json(msg='Type must be standard if either PoolA or PoolB is not specified.')
    return props