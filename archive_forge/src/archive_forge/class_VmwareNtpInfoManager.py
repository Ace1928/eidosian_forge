from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
class VmwareNtpInfoManager(PyVmomi):

    def __init__(self, module):
        super(VmwareNtpInfoManager, self).__init__(module)
        cluster_name = self.params.get('cluster_name', None)
        esxi_host_name = self.params.get('esxi_hostname', None)
        self.hosts = self.get_all_host_objs(cluster_name=cluster_name, esxi_host_name=esxi_host_name)

    def gather_ntp_info(self):
        hosts_info = {}
        for host in self.hosts:
            host_ntp_info = []
            host_date_time_manager = host.configManager.dateTimeSystem
            if host_date_time_manager:
                host_ntp_info.append(dict(time_zone_identifier=host_date_time_manager.dateTimeInfo.timeZone.key, time_zone_name=host_date_time_manager.dateTimeInfo.timeZone.name, time_zone_description=host_date_time_manager.dateTimeInfo.timeZone.description, time_zone_gmt_offset=host_date_time_manager.dateTimeInfo.timeZone.gmtOffset, ntp_servers=list(host_date_time_manager.dateTimeInfo.ntpConfig.server)))
            hosts_info[host.name] = host_ntp_info
        return hosts_info