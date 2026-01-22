from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
from ansible.module_utils._text import to_native
class VmkernelInfoManager(PyVmomi):

    def __init__(self, module):
        super(VmkernelInfoManager, self).__init__(module)
        cluster_name = self.params.get('cluster_name', None)
        esxi_host_name = self.params.get('esxi_hostname', None)
        self.hosts = self.get_all_host_objs(cluster_name=cluster_name, esxi_host_name=esxi_host_name)
        self.service_type_vmks = dict()
        self.get_all_vmks_by_service_type()

    def get_all_vmks_by_service_type(self):
        """
        Function to return information about service types and VMKernel

        """
        for host in self.hosts:
            self.service_type_vmks[host.name] = dict(vmotion=[], vsan=[], management=[], faultToleranceLogging=[])
            for service_type in self.service_type_vmks[host.name].keys():
                vmks_list = self.query_service_type_for_vmks(host, service_type)
                self.service_type_vmks[host.name][service_type] = vmks_list

    def query_service_type_for_vmks(self, host_system, service_type):
        """
        Function to return list of VMKernels
        Args:
            host_system: Host system managed object
            service_type: Name of service type

        Returns: List of VMKernel which belongs to that service type

        """
        vmks_list = []
        query = None
        try:
            query = host_system.configManager.virtualNicManager.QueryNetConfig(service_type)
        except vim.fault.HostConfigFault as config_fault:
            self.module.fail_json(msg='Failed to get all VMKs for service type %s due to host config fault : %s' % (service_type, to_native(config_fault.msg)))
        except vmodl.fault.InvalidArgument as invalid_argument:
            self.module.fail_json(msg='Failed to get all VMKs for service type %s due to invalid arguments : %s' % (service_type, to_native(invalid_argument.msg)))
        except Exception as e:
            self.module.fail_json(msg='Failed to get all VMKs for service type %s due to%s' % (service_type, to_native(e)))
        if not query or not query.selectedVnic:
            return vmks_list
        selected_vnics = list(query.selectedVnic)
        vnics_with_service_type = [vnic.device for vnic in query.candidateVnic if vnic.key in selected_vnics]
        return vnics_with_service_type

    def gather_host_vmk_info(self):
        hosts_info = {}
        for host in self.hosts:
            host_vmk_info = []
            host_network_system = host.config.network
            if host_network_system:
                vmks_config = host.config.network.vnic
                for vmk in vmks_config:
                    host_vmk_info.append(dict(device=vmk.device, key=vmk.key, portgroup=vmk.portgroup, ipv4_address=vmk.spec.ip.ipAddress, ipv4_subnet_mask=vmk.spec.ip.subnetMask, dhcp=vmk.spec.ip.dhcp, mac=vmk.spec.mac, mtu=vmk.spec.mtu, stack=vmk.spec.netStackInstanceKey, enable_vsan=vmk.device in self.service_type_vmks[host.name]['vsan'], enable_vmotion=vmk.device in self.service_type_vmks[host.name]['vmotion'], enable_management=vmk.device in self.service_type_vmks[host.name]['management'], enable_ft=vmk.device in self.service_type_vmks[host.name]['faultToleranceLogging']))
            hosts_info[host.name] = host_vmk_info
        return hosts_info