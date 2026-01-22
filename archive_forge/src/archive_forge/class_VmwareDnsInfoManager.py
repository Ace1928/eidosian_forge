from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
class VmwareDnsInfoManager(PyVmomi):

    def __init__(self, module):
        super(VmwareDnsInfoManager, self).__init__(module)
        cluster_name = self.params.get('cluster_name', None)
        esxi_host_name = self.params.get('esxi_hostname', None)
        self.hosts = self.get_all_host_objs(cluster_name=cluster_name, esxi_host_name=esxi_host_name)

    def gather_dns_info(self):
        hosts_info = {}
        for host in self.hosts:
            host_info = {}
            dns_config = host.config.network.dnsConfig
            host_info['dhcp'] = dns_config.dhcp
            host_info['virtual_nic_device'] = dns_config.virtualNicDevice
            host_info['host_name'] = dns_config.hostName
            host_info['domain_name'] = dns_config.domainName
            host_info['ip_address'] = list(dns_config.address)
            host_info['search_domain'] = list(dns_config.searchDomain)
            hosts_info[host.name] = host_info
        return hosts_info