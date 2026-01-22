from __future__ import (absolute_import, division, print_function)
import glob
import os
import re
from ansible.module_utils.facts.virtual.base import Virtual, VirtualCollector
from ansible.module_utils.facts.utils import get_file_content, get_file_lines
class LinuxVirtual(Virtual):
    """
    This is a Linux-specific subclass of Virtual.  It defines
    - virtualization_type
    - virtualization_role
    """
    platform = 'Linux'

    def get_virtual_facts(self):
        virtual_facts = {}
        found_virt = False
        host_tech = set()
        guest_tech = set()
        if os.path.exists('/proc/1/cgroup'):
            for line in get_file_lines('/proc/1/cgroup'):
                if re.search('/docker(/|-[0-9a-f]+\\.scope)', line):
                    guest_tech.add('docker')
                    if not found_virt:
                        virtual_facts['virtualization_type'] = 'docker'
                        virtual_facts['virtualization_role'] = 'guest'
                        found_virt = True
                if re.search('/lxc/', line) or re.search('/machine.slice/machine-lxc', line):
                    guest_tech.add('lxc')
                    if not found_virt:
                        virtual_facts['virtualization_type'] = 'lxc'
                        virtual_facts['virtualization_role'] = 'guest'
                        found_virt = True
                if re.search('/system.slice/containerd.service', line):
                    guest_tech.add('containerd')
                    if not found_virt:
                        virtual_facts['virtualization_type'] = 'containerd'
                        virtual_facts['virtualization_role'] = 'guest'
                        found_virt = True
        if os.path.exists('/proc/1/environ'):
            for line in get_file_lines('/proc/1/environ', line_sep='\x00'):
                if re.search('container=lxc', line):
                    guest_tech.add('lxc')
                    if not found_virt:
                        virtual_facts['virtualization_type'] = 'lxc'
                        virtual_facts['virtualization_role'] = 'guest'
                        found_virt = True
                if re.search('container=podman', line):
                    guest_tech.add('podman')
                    if not found_virt:
                        virtual_facts['virtualization_type'] = 'podman'
                        virtual_facts['virtualization_role'] = 'guest'
                        found_virt = True
                if re.search('^container=.', line):
                    guest_tech.add('container')
                    if not found_virt:
                        virtual_facts['virtualization_type'] = 'container'
                        virtual_facts['virtualization_role'] = 'guest'
                        found_virt = True
        if os.path.exists('/proc/vz') and (not os.path.exists('/proc/lve')):
            virtual_facts['virtualization_type'] = 'openvz'
            if os.path.exists('/proc/bc'):
                host_tech.add('openvz')
                if not found_virt:
                    virtual_facts['virtualization_role'] = 'host'
            else:
                guest_tech.add('openvz')
                if not found_virt:
                    virtual_facts['virtualization_role'] = 'guest'
            found_virt = True
        systemd_container = get_file_content('/run/systemd/container')
        if systemd_container:
            guest_tech.add(systemd_container)
            if not found_virt:
                virtual_facts['virtualization_type'] = systemd_container
                virtual_facts['virtualization_role'] = 'guest'
                found_virt = True
        if os.path.exists('/.dockerenv') or os.path.exists('/.dockerinit'):
            guest_tech.add('docker')
            if not found_virt:
                virtual_facts['virtualization_type'] = 'docker'
                virtual_facts['virtualization_role'] = 'guest'
                found_virt = True
        if guest_tech.intersection(set(['docker', 'lxc', 'podman', 'openvz', 'containerd'])) or systemd_container:
            guest_tech.add('container')
        if os.path.exists('/proc/xen'):
            is_xen_host = False
            try:
                for line in get_file_lines('/proc/xen/capabilities'):
                    if 'control_d' in line:
                        is_xen_host = True
            except IOError:
                pass
            if is_xen_host:
                host_tech.add('xen')
                if not found_virt:
                    virtual_facts['virtualization_type'] = 'xen'
                    virtual_facts['virtualization_role'] = 'host'
            elif not found_virt:
                virtual_facts['virtualization_type'] = 'xen'
                virtual_facts['virtualization_role'] = 'guest'
            found_virt = True
        if not found_virt:
            virtual_facts['virtualization_role'] = 'guest'
        product_name = get_file_content('/sys/devices/virtual/dmi/id/product_name')
        sys_vendor = get_file_content('/sys/devices/virtual/dmi/id/sys_vendor')
        product_family = get_file_content('/sys/devices/virtual/dmi/id/product_family')
        if product_name in ('KVM', 'KVM Server', 'Bochs', 'AHV'):
            guest_tech.add('kvm')
            if not found_virt:
                virtual_facts['virtualization_type'] = 'kvm'
                found_virt = True
        if sys_vendor == 'oVirt':
            guest_tech.add('oVirt')
            if not found_virt:
                virtual_facts['virtualization_type'] = 'oVirt'
                found_virt = True
        if sys_vendor == 'Red Hat':
            if product_family == 'RHV':
                guest_tech.add('RHV')
                if not found_virt:
                    virtual_facts['virtualization_type'] = 'RHV'
                    found_virt = True
            elif product_name == 'RHEV Hypervisor':
                guest_tech.add('RHEV')
                if not found_virt:
                    virtual_facts['virtualization_type'] = 'RHEV'
                    found_virt = True
        if product_name and product_name.startswith(('VMware',)):
            guest_tech.add('VMware')
            if not found_virt:
                virtual_facts['virtualization_type'] = 'VMware'
                found_virt = True
        if product_name in ('OpenStack Compute', 'OpenStack Nova'):
            guest_tech.add('openstack')
            if not found_virt:
                virtual_facts['virtualization_type'] = 'openstack'
                found_virt = True
        bios_vendor = get_file_content('/sys/devices/virtual/dmi/id/bios_vendor')
        if bios_vendor == 'Xen':
            guest_tech.add('xen')
            if not found_virt:
                virtual_facts['virtualization_type'] = 'xen'
                found_virt = True
        if bios_vendor == 'innotek GmbH':
            guest_tech.add('virtualbox')
            if not found_virt:
                virtual_facts['virtualization_type'] = 'virtualbox'
                found_virt = True
        if bios_vendor in ('Amazon EC2', 'DigitalOcean', 'Hetzner'):
            guest_tech.add('kvm')
            if not found_virt:
                virtual_facts['virtualization_type'] = 'kvm'
                found_virt = True
        KVM_SYS_VENDORS = ('QEMU', 'Amazon EC2', 'DigitalOcean', 'Google', 'Scaleway', 'Nutanix')
        if sys_vendor in KVM_SYS_VENDORS:
            guest_tech.add('kvm')
            if not found_virt:
                virtual_facts['virtualization_type'] = 'kvm'
                found_virt = True
        if sys_vendor == 'KubeVirt':
            guest_tech.add('KubeVirt')
            if not found_virt:
                virtual_facts['virtualization_type'] = 'KubeVirt'
                found_virt = True
        if sys_vendor == 'Microsoft Corporation':
            guest_tech.add('VirtualPC')
            if not found_virt:
                virtual_facts['virtualization_type'] = 'VirtualPC'
                found_virt = True
        if sys_vendor == 'Parallels Software International Inc.':
            guest_tech.add('parallels')
            if not found_virt:
                virtual_facts['virtualization_type'] = 'parallels'
                found_virt = True
        if sys_vendor == 'OpenStack Foundation':
            guest_tech.add('openstack')
            if not found_virt:
                virtual_facts['virtualization_type'] = 'openstack'
                found_virt = True
        if not found_virt:
            del virtual_facts['virtualization_role']
        if os.path.exists('/proc/self/status'):
            for line in get_file_lines('/proc/self/status'):
                if re.match('^VxID:\\s+\\d+', line):
                    if not found_virt:
                        virtual_facts['virtualization_type'] = 'linux_vserver'
                    if re.match('^VxID:\\s+0', line):
                        host_tech.add('linux_vserver')
                        if not found_virt:
                            virtual_facts['virtualization_role'] = 'host'
                    else:
                        guest_tech.add('linux_vserver')
                        if not found_virt:
                            virtual_facts['virtualization_role'] = 'guest'
                    found_virt = True
        if os.path.exists('/proc/cpuinfo'):
            for line in get_file_lines('/proc/cpuinfo'):
                if re.match('^model name.*QEMU Virtual CPU', line):
                    guest_tech.add('kvm')
                    if not found_virt:
                        virtual_facts['virtualization_type'] = 'kvm'
                elif re.match('^vendor_id.*User Mode Linux', line):
                    guest_tech.add('uml')
                    if not found_virt:
                        virtual_facts['virtualization_type'] = 'uml'
                elif re.match('^model name.*UML', line):
                    guest_tech.add('uml')
                    if not found_virt:
                        virtual_facts['virtualization_type'] = 'uml'
                elif re.match('^machine.*CHRP IBM pSeries .emulated by qemu.', line):
                    guest_tech.add('kvm')
                    if not found_virt:
                        virtual_facts['virtualization_type'] = 'kvm'
                elif re.match('^vendor_id.*PowerVM Lx86', line):
                    guest_tech.add('powervm_lx86')
                    if not found_virt:
                        virtual_facts['virtualization_type'] = 'powervm_lx86'
                elif re.match('^vendor_id.*IBM/S390', line):
                    guest_tech.add('PR/SM')
                    if not found_virt:
                        virtual_facts['virtualization_type'] = 'PR/SM'
                    lscpu = self.module.get_bin_path('lscpu')
                    if lscpu:
                        rc, out, err = self.module.run_command(['lscpu'])
                        if rc == 0:
                            for line in out.splitlines():
                                data = line.split(':', 1)
                                key = data[0].strip()
                                if key == 'Hypervisor':
                                    tech = data[1].strip()
                                    guest_tech.add(tech)
                                    if not found_virt:
                                        virtual_facts['virtualization_type'] = tech
                    else:
                        guest_tech.add('ibm_systemz')
                        if not found_virt:
                            virtual_facts['virtualization_type'] = 'ibm_systemz'
                else:
                    continue
                if virtual_facts['virtualization_type'] == 'PR/SM':
                    if not found_virt:
                        virtual_facts['virtualization_role'] = 'LPAR'
                elif not found_virt:
                    virtual_facts['virtualization_role'] = 'guest'
                if not found_virt:
                    found_virt = True
        if os.path.exists('/proc/modules') and os.access('/proc/modules', os.R_OK):
            modules = []
            for line in get_file_lines('/proc/modules'):
                data = line.split(' ', 1)
                modules.append(data[0])
            if 'kvm' in modules:
                host_tech.add('kvm')
                if not found_virt:
                    virtual_facts['virtualization_type'] = 'kvm'
                    virtual_facts['virtualization_role'] = 'host'
                if os.path.isdir('/rhev/'):
                    for f in glob.glob('/proc/[0-9]*/comm'):
                        try:
                            with open(f) as virt_fh:
                                comm_content = virt_fh.read().rstrip()
                            if comm_content in ('vdsm', 'vdsmd'):
                                host_tech.add('RHEV')
                                if not found_virt:
                                    virtual_facts['virtualization_type'] = 'RHEV'
                                break
                        except Exception:
                            pass
                found_virt = True
            if 'vboxdrv' in modules:
                host_tech.add('virtualbox')
                if not found_virt:
                    virtual_facts['virtualization_type'] = 'virtualbox'
                    virtual_facts['virtualization_role'] = 'host'
                    found_virt = True
            if 'virtio' in modules:
                host_tech.add('kvm')
                if not found_virt:
                    virtual_facts['virtualization_type'] = 'kvm'
                    virtual_facts['virtualization_role'] = 'guest'
                    found_virt = True
        dmi_bin = self.module.get_bin_path('dmidecode')
        if dmi_bin is not None:
            rc, out, err = self.module.run_command('%s -s system-product-name' % dmi_bin)
            if rc == 0:
                vendor_name = ''.join([line.strip() for line in out.splitlines() if not line.startswith('#')])
                if vendor_name.startswith('VMware'):
                    guest_tech.add('VMware')
                    if not found_virt:
                        virtual_facts['virtualization_type'] = 'VMware'
                        virtual_facts['virtualization_role'] = 'guest'
                        found_virt = True
                if 'BHYVE' in out:
                    guest_tech.add('bhyve')
                    if not found_virt:
                        virtual_facts['virtualization_type'] = 'bhyve'
                        virtual_facts['virtualization_role'] = 'guest'
                        found_virt = True
        if os.path.exists('/dev/kvm'):
            host_tech.add('kvm')
            if not found_virt:
                virtual_facts['virtualization_type'] = 'kvm'
                virtual_facts['virtualization_role'] = 'host'
                found_virt = True
        if not found_virt:
            virtual_facts['virtualization_type'] = 'NA'
            virtual_facts['virtualization_role'] = 'NA'
            found_virt = True
        virtual_facts['virtualization_tech_guest'] = guest_tech
        virtual_facts['virtualization_tech_host'] = host_tech
        return virtual_facts