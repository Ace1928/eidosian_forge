from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_text
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
class VmwareCustomSpecManger(PyVmomi):

    def __init__(self, module):
        super(VmwareCustomSpecManger, self).__init__(module)
        self.cc_mgr = self.content.customizationSpecManager
        if self.cc_mgr is None:
            self.module.fail_json(msg='Failed to get customization spec manager.')

    def gather_custom_spec_info(self):
        """
        Gather information about customization specifications
        """
        spec_name = self.params.get('spec_name', None)
        specs_list = []
        if spec_name:
            if self.cc_mgr.DoesCustomizationSpecExist(name=spec_name):
                specs_list.append(spec_name)
            else:
                self.module.fail_json(msg="Unable to find customization specification named '%s'" % spec_name)
        else:
            available_specs = self.cc_mgr.info
            for spec_info in available_specs:
                specs_list.append(spec_info.name)
        spec_info = dict()
        for spec in specs_list:
            current_spec = self.cc_mgr.GetCustomizationSpec(name=spec)
            adapter_mapping_list = []
            for nic in current_spec.spec.nicSettingMap:
                temp_data = dict(mac_address=nic.macAddress, ip_address=nic.adapter.ip.ipAddress if hasattr(nic.adapter.ip, 'ipAddress') else None, subnet_mask=nic.adapter.subnetMask, gateway=list(nic.adapter.gateway), nic_dns_server_list=list(nic.adapter.dnsServerList), dns_domain=nic.adapter.dnsDomain, primary_wins=nic.adapter.primaryWINS, secondry_wins=nic.adapter.secondaryWINS, net_bios=nic.adapter.netBIOS)
                adapter_mapping_list.append(temp_data)
            current_hostname = None
            domain = None
            time_zone = None
            hw_clock = None
            if isinstance(current_spec.spec.identity, vim.vm.customization.LinuxPrep):
                if isinstance(current_spec.spec.identity.hostName, vim.vm.customization.PrefixNameGenerator):
                    current_hostname = current_spec.spec.identity.hostName.base
                elif isinstance(current_spec.spec.identity.hostName, vim.vm.customization.FixedName):
                    current_hostname = current_spec.spec.identity.hostName.name
                domain = current_spec.spec.identity.domain
                time_zone = current_spec.spec.identity.timeZone
                hw_clock = current_spec.spec.identity.hwClockUTC
            else:
                time_zone = current_spec.spec.identity.guiUnattended.timeZone
            spec_info[spec] = dict(name=current_spec.info.name, description=current_spec.info.description, type=current_spec.info.type, last_updated_time=current_spec.info.lastUpdateTime, change_version=current_spec.info.changeVersion, hostname=current_hostname, domain=domain, time_zone=time_zone, hw_clock_utc=hw_clock, dns_suffix_list=list(current_spec.spec.globalIPSettings.dnsSuffixList), dns_server_list=list(current_spec.spec.globalIPSettings.dnsServerList), nic_setting_map=adapter_mapping_list)
        return spec_info