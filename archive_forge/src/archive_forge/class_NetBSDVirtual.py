from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.facts.virtual.base import Virtual, VirtualCollector
from ansible.module_utils.facts.virtual.sysctl import VirtualSysctlDetectionMixin
class NetBSDVirtual(Virtual, VirtualSysctlDetectionMixin):
    platform = 'NetBSD'

    def get_virtual_facts(self):
        virtual_facts = {}
        host_tech = set()
        guest_tech = set()
        virtual_facts['virtualization_type'] = ''
        virtual_facts['virtualization_role'] = ''
        virtual_product_facts = self.detect_virt_product('machdep.dmi.system-product')
        guest_tech.update(virtual_product_facts['virtualization_tech_guest'])
        host_tech.update(virtual_product_facts['virtualization_tech_host'])
        virtual_facts.update(virtual_product_facts)
        virtual_vendor_facts = self.detect_virt_vendor('machdep.dmi.system-vendor')
        guest_tech.update(virtual_vendor_facts['virtualization_tech_guest'])
        host_tech.update(virtual_vendor_facts['virtualization_tech_host'])
        if virtual_facts['virtualization_type'] == '':
            virtual_facts.update(virtual_vendor_facts)
        virtual_vendor_facts = self.detect_virt_vendor('machdep.hypervisor')
        guest_tech.update(virtual_vendor_facts['virtualization_tech_guest'])
        host_tech.update(virtual_vendor_facts['virtualization_tech_host'])
        if virtual_facts['virtualization_type'] == '':
            virtual_facts.update(virtual_vendor_facts)
        if os.path.exists('/dev/xencons'):
            guest_tech.add('xen')
            if virtual_facts['virtualization_type'] == '':
                virtual_facts['virtualization_type'] = 'xen'
                virtual_facts['virtualization_role'] = 'guest'
        virtual_facts['virtualization_tech_guest'] = guest_tech
        virtual_facts['virtualization_tech_host'] = host_tech
        return virtual_facts