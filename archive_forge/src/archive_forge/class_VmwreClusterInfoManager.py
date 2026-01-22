from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import unquote
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, find_datacenter_by_name, find_cluster_by_name, \
from ansible_collections.community.vmware.plugins.module_utils.vmware_rest_client import VmwareRestClient
class VmwreClusterInfoManager(PyVmomi):

    def __init__(self, module):
        super(VmwreClusterInfoManager, self).__init__(module)
        datacenter = self.params.get('datacenter')
        cluster_name = self.params.get('cluster_name')
        self.schema = self.params.get('schema')
        self.properties = self.params.get('properties')
        self.cluster_objs = []
        if datacenter:
            datacenter_obj = find_datacenter_by_name(self.content, datacenter_name=datacenter)
            if datacenter_obj is None:
                self.module.fail_json(msg="Failed to find datacenter '%s'" % datacenter)
            self.cluster_objs = self.get_all_cluster_objs(parent=datacenter_obj)
        elif cluster_name:
            cluster_obj = find_cluster_by_name(self.content, cluster_name=cluster_name)
            if cluster_obj is None:
                self.module.fail_json(msg="Failed to find cluster '%s'" % cluster_name)
            self.cluster_objs = [cluster_obj]

    def get_all_cluster_objs(self, parent):
        """
        Get all cluster managed objects from given parent object
        Args:
            parent: Managed objected of datacenter or host folder

        Returns: List of host managed objects

        """
        cluster_objs = []
        if isinstance(parent, vim.Datacenter):
            folder = parent.hostFolder
        else:
            folder = parent
        for child in folder.childEntity:
            if isinstance(child, vim.Folder):
                cluster_objs = cluster_objs + self.get_all_cluster_objs(child)
            if isinstance(child, vim.ClusterComputeResource):
                cluster_objs.append(child)
        return cluster_objs

    def gather_cluster_info(self):
        """
        Gather information about cluster
        """
        results = dict(changed=False, clusters=dict())
        if self.schema == 'summary':
            for cluster in self.cluster_objs:
                ha_failover_level = None
                ha_restart_priority = None
                ha_vm_tools_monitoring = None
                ha_vm_min_up_time = None
                ha_vm_max_failures = None
                ha_vm_max_failure_window = None
                ha_vm_failure_interval = None
                enabled_vsan = False
                vsan_auto_claim_storage = False
                hosts = []
                for host in cluster.host:
                    hosts.append({'name': host.name, 'folder': self.get_vm_path(self.content, host)})
                das_config = cluster.configurationEx.dasConfig
                if das_config.admissionControlPolicy:
                    ha_failover_level = das_config.admissionControlPolicy.failoverLevel
                if das_config.defaultVmSettings:
                    ha_restart_priority = das_config.defaultVmSettings.restartPriority
                    ha_vm_tools_monitoring = das_config.defaultVmSettings.vmToolsMonitoringSettings.vmMonitoring
                    ha_vm_min_up_time = das_config.defaultVmSettings.vmToolsMonitoringSettings.minUpTime
                    ha_vm_max_failures = das_config.defaultVmSettings.vmToolsMonitoringSettings.maxFailures
                    ha_vm_max_failure_window = das_config.defaultVmSettings.vmToolsMonitoringSettings.maxFailureWindow
                    ha_vm_failure_interval = das_config.defaultVmSettings.vmToolsMonitoringSettings.failureInterval
                drs_config = cluster.configurationEx.drsConfig
                if hasattr(cluster.configurationEx, 'vsanConfigInfo'):
                    vsan_config = cluster.configurationEx.vsanConfigInfo
                    enabled_vsan = vsan_config.enabled
                    vsan_auto_claim_storage = vsan_config.defaultConfig.autoClaimStorage
                tag_info = []
                if self.params.get('show_tag'):
                    vmware_client = VmwareRestClient(self.module)
                    tag_info = vmware_client.get_tags_for_cluster(cluster_mid=cluster._moId)
                resource_summary = self.to_json(cluster.GetResourceUsage())
                if '_vimtype' in resource_summary:
                    del resource_summary['_vimtype']
                results['clusters'][unquote(cluster.name)] = dict(hosts=hosts, enable_ha=das_config.enabled, ha_failover_level=ha_failover_level, ha_vm_monitoring=das_config.vmMonitoring, ha_host_monitoring=das_config.hostMonitoring, ha_admission_control_enabled=das_config.admissionControlEnabled, ha_restart_priority=ha_restart_priority, ha_vm_tools_monitoring=ha_vm_tools_monitoring, ha_vm_min_up_time=ha_vm_min_up_time, ha_vm_max_failures=ha_vm_max_failures, ha_vm_max_failure_window=ha_vm_max_failure_window, ha_vm_failure_interval=ha_vm_failure_interval, enabled_drs=drs_config.enabled, drs_enable_vm_behavior_overrides=drs_config.enableVmBehaviorOverrides, drs_default_vm_behavior=drs_config.defaultVmBehavior, drs_vmotion_rate=drs_config.vmotionRate, enabled_vsan=enabled_vsan, vsan_auto_claim_storage=vsan_auto_claim_storage, tags=tag_info, resource_summary=resource_summary, moid=cluster._moId, datacenter=get_parent_datacenter(cluster).name)
        else:
            for cluster in self.cluster_objs:
                results['clusters'][unquote(cluster.name)] = self.to_json(cluster, self.properties)
        self.module.exit_json(**results)