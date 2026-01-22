from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
from ansible.module_utils._text import to_native
class VmwareHostDNS(PyVmomi):
    """Class to manage DNS configuration of an ESXi host system"""

    def __init__(self, module):
        super(VmwareHostDNS, self).__init__(module)
        self.cluster_name = self.params.get('cluster_name')
        self.esxi_host_name = self.params.get('esxi_hostname')
        if self.is_vcenter():
            if not self.cluster_name and (not self.esxi_host_name):
                self.module.fail_json(msg="You connected to a vCenter but didn't specify the cluster_name or esxi_hostname you want to configure.")
        else:
            if self.cluster_name:
                self.module.warn('You connected directly to an ESXi host, cluster_name will be ignored.')
            if self.esxi_host_name:
                self.module.warn('You connected directly to an ESXi host, esxi_host_name will be ignored.')
        self.hosts = self.get_all_host_objs(cluster_name=self.cluster_name, esxi_host_name=self.esxi_host_name)
        if not self.hosts:
            self.module.fail_json(msg='Failed to find host system(s).')
        self.network_type = self.params.get('type')
        self.vmkernel_device = self.params.get('device')
        self.host_name = self.params.get('host_name')
        self.domain = self.params.get('domain')
        self.dns_servers = self.params.get('dns_servers')
        self.search_domains = self.params.get('search_domains')

    def ensure(self):
        """Function to manage DNS configuration of an ESXi host system"""
        results = dict(changed=False, dns_config_result=dict())
        verbose = self.module.params.get('verbose', False)
        host_change_list = []
        for host in self.hosts:
            initial_name = host.name
            changed = False
            changed_list = []
            host_result = {'changed': '', 'msg': '', 'host_name': host.name}
            host_netstack_config = host.config.network.netStackInstance
            for instance in host_netstack_config:
                if instance.key == 'defaultTcpipStack':
                    netstack_spec = vim.host.NetworkConfig.NetStackSpec()
                    netstack_spec.operation = 'edit'
                    netstack_spec.netStackInstance = vim.host.NetStackInstance()
                    netstack_spec.netStackInstance.key = 'defaultTcpipStack'
                    dns_config = vim.host.DnsConfig()
                    host_result['dns_config'] = self.network_type
                    host_result['search_domains'] = self.search_domains
                    if self.network_type == 'static':
                        if self.host_name:
                            if instance.dnsConfig.hostName != self.host_name:
                                host_result['host_name_previous'] = instance.dnsConfig.hostName
                                changed = True
                                changed_list.append('Host name')
                            dns_config.hostName = self.host_name
                        else:
                            dns_config.hostName = instance.dnsConfig.hostName
                        if self.search_domains is not None:
                            if instance.dnsConfig.searchDomain != self.search_domains:
                                host_result['search_domains_previous'] = instance.dnsConfig.searchDomain
                                host_result['search_domains_changed'] = self.get_differt_entries(instance.dnsConfig.searchDomain, self.search_domains)
                                changed = True
                                changed_list.append('Search domains')
                            dns_config.searchDomain = self.search_domains
                        else:
                            dns_config.searchDomain = instance.dnsConfig.searchDomain
                        if instance.dnsConfig.dhcp:
                            host_result['domain'] = self.domain
                            host_result['dns_servers'] = self.dns_servers
                            host_result['search_domains'] = self.search_domains
                            host_result['dns_config_previous'] = 'DHCP'
                            changed = True
                            changed_list.append('DNS configuration')
                            dns_config.dhcp = False
                            dns_config.virtualNicDevice = None
                            dns_config.domainName = self.domain
                            dns_config.address = self.dns_servers
                            dns_config.searchDomain = self.search_domains
                        else:
                            host_result['domain'] = self.domain
                            if self.domain is not None:
                                if instance.dnsConfig.domainName != self.domain:
                                    host_result['domain_previous'] = instance.dnsConfig.domainName
                                    changed = True
                                    changed_list.append('Domain')
                                dns_config.domainName = self.domain
                            else:
                                dns_config.domainName = instance.dnsConfig.domainName
                            host_result['dns_servers'] = self.dns_servers
                            if self.dns_servers is not None:
                                if instance.dnsConfig.address != self.dns_servers:
                                    host_result['dns_servers_previous'] = instance.dnsConfig.address
                                    host_result['dns_servers_changed'] = self.get_differt_entries(instance.dnsConfig.address, self.dns_servers)
                                    changed = True
                                    if verbose:
                                        dns_servers_verbose_message = self.build_changed_message(instance.dnsConfig.address, self.dns_servers)
                                    else:
                                        changed_list.append('DNS servers')
                                dns_config.address = self.dns_servers
                            else:
                                dns_config.address = instance.dnsConfig.address
                    elif self.network_type == 'dhcp' and (not instance.dnsConfig.dhcp):
                        host_result['device'] = self.vmkernel_device
                        host_result['dns_config_previous'] = 'static'
                        changed = True
                        changed_list.append('DNS configuration')
                        dns_config.dhcp = True
                        dns_config.virtualNicDevice = self.vmkernel_device
                    netstack_spec.netStackInstance.dnsConfig = dns_config
                    config = vim.host.NetworkConfig()
                    config.netStackSpec = [netstack_spec]
            if changed:
                if self.module.check_mode:
                    changed_suffix = ' would be changed'
                else:
                    changed_suffix = ' changed'
                if len(changed_list) > 2:
                    message = ', '.join(changed_list[:-1]) + ', and ' + str(changed_list[-1])
                elif len(changed_list) == 2:
                    message = ' and '.join(changed_list)
                elif len(changed_list) == 1:
                    message = changed_list[0]
                if verbose and dns_servers_verbose_message:
                    if changed_list:
                        message = message + changed_suffix + '. ' + dns_servers_verbose_message + '.'
                    else:
                        message = dns_servers_verbose_message
                else:
                    message += changed_suffix
                host_result['changed'] = True
                host_network_system = host.configManager.networkSystem
                if not self.module.check_mode:
                    try:
                        host_network_system.UpdateNetworkConfig(config, 'modify')
                    except vim.fault.AlreadyExists:
                        self.module.fail_json(msg="Network entity specified in the configuration already exist on host '%s'" % host.name)
                    except vim.fault.NotFound:
                        self.module.fail_json(msg="Network entity specified in the configuration doesn't exist on host '%s'" % host.name)
                    except vim.fault.ResourceInUse:
                        self.module.fail_json(msg="Resource is in use on host '%s'" % host.name)
                    except vmodl.fault.InvalidArgument:
                        self.module.fail_json(msg="An invalid parameter is passed in for one of the networking objects for host '%s'" % host.name)
                    except vmodl.fault.NotSupported as not_supported:
                        self.module.fail_json(msg="Operation isn't supported for the instance on '%s' : %s" % (host.name, to_native(not_supported.msg)))
                    except vim.fault.HostConfigFault as config_fault:
                        self.module.fail_json(msg="Failed to configure TCP/IP stacks for host '%s' due to : %s" % (host.name, to_native(config_fault.msg)))
            else:
                host_result['changed'] = False
                message = 'All settings are already configured'
            host_result['msg'] = message
            results['dns_config_result'][initial_name] = host_result
            host_change_list.append(changed)
        if any(host_change_list):
            results['changed'] = True
        self.module.exit_json(**results)

    def build_changed_message(self, dns_servers_configured, dns_servers_new):
        """Build changed message"""
        check_mode = 'would be ' if self.module.check_mode else ''
        add = self.get_not_in_list_one(dns_servers_new, dns_servers_configured)
        remove = self.get_not_in_list_one(dns_servers_configured, dns_servers_new)
        diff_servers = list(dns_servers_configured)
        if add and remove:
            for server in add:
                diff_servers.append(server)
            for server in remove:
                diff_servers.remove(server)
            if dns_servers_new != diff_servers:
                message = 'DNS server %s %sadded and %s %sremoved and the server sequence %schanged as well' % (self.array_to_string(add), check_mode, self.array_to_string(remove), check_mode, check_mode)
            elif dns_servers_new != dns_servers_configured:
                message = 'DNS server %s %sreplaced with %s' % (self.array_to_string(remove), check_mode, self.array_to_string(add))
            else:
                message = 'DNS server %s %sremoved and %s %sadded' % (self.array_to_string(remove), check_mode, self.array_to_string(add), check_mode)
        elif add:
            for server in add:
                diff_servers.append(server)
            if dns_servers_new != diff_servers:
                message = 'DNS server %s %sadded and the server sequence %schanged as well' % (self.array_to_string(add), check_mode, check_mode)
            else:
                message = 'DNS server %s %sadded' % (self.array_to_string(add), check_mode)
        elif remove:
            for server in remove:
                diff_servers.remove(server)
            if dns_servers_new != diff_servers:
                message = 'DNS server %s %sremoved and the server sequence %schanged as well' % (self.array_to_string(remove), check_mode, check_mode)
            else:
                message = 'DNS server %s %sremoved' % (self.array_to_string(remove), check_mode)
        else:
            message = 'DNS server sequence %schanged' % check_mode
        return message

    @staticmethod
    def get_not_in_list_one(list1, list2):
        """Return entries that ore not in list one"""
        return [x for x in list1 if x not in set(list2)]

    @staticmethod
    def array_to_string(array):
        """Return string from array"""
        if len(array) > 2:
            string = ', '.join(("'{0}'".format(element) for element in array[:-1])) + ', and ' + "'{0}'".format(str(array[-1]))
        elif len(array) == 2:
            string = ' and '.join(("'{0}'".format(element) for element in array))
        elif len(array) == 1:
            string = "'{0}'".format(array[0])
        return string

    @staticmethod
    def get_differt_entries(list1, list2):
        """Return different entries of two lists"""
        return [a for a in list1 + list2 if a not in list1 or a not in list2]