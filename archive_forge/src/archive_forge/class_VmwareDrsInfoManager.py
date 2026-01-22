from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi, find_datacenter_by_name, get_all_objs
class VmwareDrsInfoManager(PyVmomi):

    def __init__(self, module):
        super(VmwareDrsInfoManager, self).__init__(module)
        datacenter_name = self.params.get('datacenter', None)
        if datacenter_name:
            datacenter_obj = find_datacenter_by_name(self.content, datacenter_name=datacenter_name)
            self.cluster_obj_list = []
            if datacenter_obj:
                folder = datacenter_obj.hostFolder
                self.cluster_obj_list = get_all_objs(self.content, [vim.ClusterComputeResource], folder)
            else:
                self.module.fail_json(changed=False, msg="Datacenter '%s' not found" % datacenter_name)
        cluster_name = self.params.get('cluster_name', None)
        if cluster_name:
            cluster_obj = self.find_cluster_by_name(cluster_name=cluster_name)
            if cluster_obj is None:
                self.module.fail_json(changed=False, msg="Cluster '%s' not found" % cluster_name)
            else:
                self.cluster_obj_list = [cluster_obj]

    def get_all_from_group(self, group_name=None, cluster_obj=None, hostgroup=False):
        """
        Return all VM / Host names using given group name
        Args:
            group_name: Rule name
            cluster_obj: Cluster managed object
            hostgroup: True if we want only host name from group

        Returns: List of VM / Host names belonging to given group object

        """
        obj_name_list = []
        if not all([group_name, cluster_obj]):
            return obj_name_list
        for group in cluster_obj.configurationEx.group:
            if group.name == group_name:
                if not hostgroup and isinstance(group, vim.cluster.VmGroup):
                    obj_name_list = [vm.name for vm in group.vm]
                    break
                if hostgroup and isinstance(group, vim.cluster.HostGroup):
                    obj_name_list = [host.name for host in group.host]
                    break
        return obj_name_list

    @staticmethod
    def normalize_vm_vm_rule_spec(rule_obj=None):
        """
        Return human readable rule spec
        Args:
            rule_obj: Rule managed object

        Returns: Dictionary with DRS VM VM Rule info

        """
        if rule_obj is None:
            return {}
        return dict(rule_key=rule_obj.key, rule_enabled=rule_obj.enabled, rule_name=rule_obj.name, rule_mandatory=rule_obj.mandatory, rule_uuid=rule_obj.ruleUuid, rule_vms=[vm.name for vm in rule_obj.vm], rule_type='vm_vm_rule', rule_affinity=True if isinstance(rule_obj, vim.cluster.AffinityRuleSpec) else False)

    def normalize_vm_host_rule_spec(self, rule_obj=None, cluster_obj=None):
        """
        Return human readable rule spec
        Args:
            rule_obj: Rule managed object
            cluster_obj: Cluster managed object

        Returns: Dictionary with DRS VM HOST Rule info

        """
        if not all([rule_obj, cluster_obj]):
            return {}
        return dict(rule_key=rule_obj.key, rule_enabled=rule_obj.enabled, rule_name=rule_obj.name, rule_mandatory=rule_obj.mandatory, rule_uuid=rule_obj.ruleUuid, rule_vm_group_name=rule_obj.vmGroupName, rule_affine_host_group_name=rule_obj.affineHostGroupName, rule_anti_affine_host_group_name=rule_obj.antiAffineHostGroupName, rule_vms=self.get_all_from_group(group_name=rule_obj.vmGroupName, cluster_obj=cluster_obj), rule_affine_hosts=self.get_all_from_group(group_name=rule_obj.affineHostGroupName, cluster_obj=cluster_obj, hostgroup=True), rule_anti_affine_hosts=self.get_all_from_group(group_name=rule_obj.antiAffineHostGroupName, cluster_obj=cluster_obj, hostgroup=True), rule_type='vm_host_rule')

    def gather_drs_rule_info(self):
        """
        Gather DRS rule information about given cluster
        Returns: Dictionary of clusters with DRS information

        """
        cluster_rule_info = dict()
        for cluster_obj in self.cluster_obj_list:
            cluster_rule_info[cluster_obj.name] = []
            for drs_rule in cluster_obj.configuration.rule:
                if isinstance(drs_rule, vim.cluster.VmHostRuleInfo):
                    cluster_rule_info[cluster_obj.name].append(self.normalize_vm_host_rule_spec(rule_obj=drs_rule, cluster_obj=cluster_obj))
                else:
                    cluster_rule_info[cluster_obj.name].append(self.normalize_vm_vm_rule_spec(rule_obj=drs_rule))
        return cluster_rule_info