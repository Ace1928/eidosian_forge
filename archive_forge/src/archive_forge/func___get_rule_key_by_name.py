from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def __get_rule_key_by_name(self, cluster_obj=None, rule_name=None):
    """
        Function to get a specific VM-Host DRS rule key by name
        Args:
            rule_name: Name of rule
            cluster_obj: Cluster managed object

        Returns: Rule Object if found or None

        """
    if cluster_obj is None:
        cluster_obj = self.__cluster_obj
    if rule_name is None:
        rule_name = self.__rule_name
    if rule_name:
        rules_list = [rule for rule in cluster_obj.configuration.rule if rule.name == rule_name]
        if rules_list:
            return rules_list[0]
    return None