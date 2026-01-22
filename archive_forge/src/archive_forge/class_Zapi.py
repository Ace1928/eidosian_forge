from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
class Zapi(ZabbixBase):

    def __init__(self, module, zbx=None):
        super(Zapi, self).__init__(module, zbx)
        self._zapi_wrapper = self

    def check_if_action_exists(self, name):
        """Check if action exists.

        Args:
            name: Name of the action.

        Returns:
            The return value. True for success, False otherwise.

        """
        try:
            _params = {'selectOperations': 'extend', 'selectRecoveryOperations': 'extend', 'selectUpdateOperations': 'extend', 'selectFilter': 'extend', 'filter': {'name': [name]}}
            _action = self._zapi.action.get(_params)
            return _action
        except Exception as e:
            self._module.fail_json(msg="Failed to check if action '%s' exists: %s" % (name, e))

    def get_action_by_name(self, name):
        """Get action by name

        Args:
            name: Name of the action.

        Returns:
            dict: Zabbix action

        """
        try:
            action_list = self._zapi.action.get({'output': 'extend', 'filter': {'name': [name]}})
            if len(action_list) < 1:
                self._module.fail_json(msg='Action not found: %s' % name)
            else:
                return action_list[0]
        except Exception as e:
            self._module.fail_json(msg="Failed to get ID of '%s': %s" % (name, e))

    def get_host_by_host_name(self, host_name):
        """Get host by host name

        Args:
            host_name: host name.

        Returns:
            host matching host name

        """
        try:
            host_list = self._zapi.host.get({'output': 'extend', 'selectInventory': 'extend', 'filter': {'host': [host_name]}})
            if len(host_list) < 1:
                self._module.fail_json(msg='Host not found: %s' % host_name)
            else:
                return host_list[0]
        except Exception as e:
            self._module.fail_json(msg="Failed to get host '%s': %s" % (host_name, e))

    def get_hostgroup_by_hostgroup_name(self, hostgroup_name):
        """Get host group by host group name

        Args:
            hostgroup_name: host group name.

        Returns:
            host group matching host group name

        """
        try:
            hostgroup_list = self._zapi.hostgroup.get({'output': 'extend', 'filter': {'name': [hostgroup_name]}})
            if len(hostgroup_list) < 1:
                self._module.fail_json(msg='Host group not found: %s' % hostgroup_name)
            else:
                return hostgroup_list[0]
        except Exception as e:
            self._module.fail_json(msg="Failed to get host group '%s': %s" % (hostgroup_name, e))

    def get_template_by_template_name(self, template_name):
        """Get template by template name

        Args:
            template_name: template name.

        Returns:
            template matching template name

        """
        try:
            template_list = self._zapi.template.get({'output': 'extend', 'filter': {'host': [template_name]}})
            if len(template_list) < 1:
                self._module.fail_json(msg='Template not found: %s' % template_name)
            else:
                return template_list[0]
        except Exception as e:
            self._module.fail_json(msg="Failed to get template '%s': %s" % (template_name, e))

    def get_trigger_by_trigger_name(self, trigger_name):
        """Get trigger by trigger name

        Args:
            trigger_name: trigger name.

        Returns:
            trigger matching trigger name

        """
        try:
            trigger_list = self._zapi.trigger.get({'output': 'extend', 'filter': {'description': [trigger_name]}})
            if len(trigger_list) < 1:
                self._module.fail_json(msg='Trigger not found: %s' % trigger_name)
            else:
                return trigger_list[0]
        except Exception as e:
            self._module.fail_json(msg="Failed to get trigger '%s': %s" % (trigger_name, e))

    def get_discovery_rule_by_discovery_rule_name(self, discovery_rule_name):
        """Get discovery rule by discovery rule name

        Args:
            discovery_rule_name: discovery rule name.

        Returns:
            discovery rule matching discovery rule name

        """
        try:
            discovery_rule_list = self._zapi.drule.get({'output': 'extend', 'filter': {'name': [discovery_rule_name]}})
            if len(discovery_rule_list) < 1:
                self._module.fail_json(msg='Discovery rule not found: %s' % discovery_rule_name)
            else:
                return discovery_rule_list[0]
        except Exception as e:
            self._module.fail_json(msg="Failed to get discovery rule '%s': %s" % (discovery_rule_name, e))

    def get_discovery_check_by_discovery_check_name(self, discovery_check_name):
        """Get discovery check  by discovery check name

        Args:
            discovery_check_name: discovery check name.

        Returns:
            discovery check matching discovery check name

        """
        try:
            discovery_rule_name, dcheck_type = discovery_check_name.split(': ')
            dcheck_type_to_number = {'SSH': '0', 'LDAP': '1', 'SMTP': '2', 'FTP': '3', 'HTTP': '4', 'POP': '5', 'NNTP': '6', 'IMAP': '7', 'TCP': '8', 'Zabbix agent': '9', 'SNMPv1 agent': '10', 'SNMPv2 agent': '11', 'ICMP ping': '12', 'SNMPv3 agent': '13', 'HTTPS': '14', 'Telnet': '15'}
            if dcheck_type not in dcheck_type_to_number:
                self._module.fail_json(msg='Discovery check type: %s does not exist' % dcheck_type)
            discovery_rule_list = self._zapi.drule.get({'output': ['dchecks'], 'filter': {'name': [discovery_rule_name]}, 'selectDChecks': 'extend'})
            if len(discovery_rule_list) < 1:
                self._module.fail_json(msg='Discovery check not found: %s' % discovery_check_name)
            for dcheck in discovery_rule_list[0]['dchecks']:
                if dcheck_type_to_number[dcheck_type] == dcheck['type']:
                    return dcheck
            self._module.fail_json(msg='Discovery check not found: %s' % discovery_check_name)
        except Exception as e:
            self._module.fail_json(msg="Failed to get discovery check '%s': %s" % (discovery_check_name, e))

    def get_proxy_by_proxy_name(self, proxy_name):
        """Get proxy by proxy name

        Args:
            proxy_name: proxy name.

        Returns:
            proxy matching proxy name

        """
        try:
            proxy_list = self._zapi.proxy.get({'output': 'extend', 'filter': {'host': [proxy_name]}})
            if len(proxy_list) < 1:
                self._module.fail_json(msg='Proxy not found: %s' % proxy_name)
            else:
                return proxy_list[0]
        except Exception as e:
            self._module.fail_json(msg="Failed to get proxy '%s': %s" % (proxy_name, e))

    def get_mediatype_by_mediatype_name(self, mediatype_name):
        """Get mediatype by mediatype name

        Args:
            mediatype_name: mediatype name

        Returns:
            mediatype matching mediatype name

        """
        filter = {'name': [mediatype_name]}
        try:
            if str(mediatype_name).lower() == 'all':
                return '0'
            mediatype_list = self._zapi.mediatype.get({'output': 'extend', 'filter': filter})
            if len(mediatype_list) < 1:
                self._module.fail_json(msg='Media type not found: %s' % mediatype_name)
            else:
                return mediatype_list[0]['mediatypeid']
        except Exception as e:
            self._module.fail_json(msg="Failed to get mediatype '%s': %s" % (mediatype_name, e))

    def get_user_by_user_name(self, user_name):
        """Get user by user name

        Args:
            user_name: user name

        Returns:
            user matching user name

        """
        try:
            filter = {'username': [user_name]}
            user_list = self._zapi.user.get({'output': 'extend', 'filter': filter})
            if len(user_list) < 1:
                self._module.fail_json(msg='User not found: %s' % user_name)
            else:
                return user_list[0]
        except Exception as e:
            self._module.fail_json(msg="Failed to get user '%s': %s" % (user_name, e))

    def get_usergroup_by_usergroup_name(self, usergroup_name):
        """Get usergroup by usergroup name

        Args:
            usergroup_name: usergroup name

        Returns:
            usergroup matching usergroup name

        """
        try:
            usergroup_list = self._zapi.usergroup.get({'output': 'extend', 'filter': {'name': [usergroup_name]}})
            if len(usergroup_list) < 1:
                self._module.fail_json(msg='User group not found: %s' % usergroup_name)
            else:
                return usergroup_list[0]
        except Exception as e:
            self._module.fail_json(msg="Failed to get user group '%s': %s" % (usergroup_name, e))

    def get_script_by_script_name(self, script_name):
        """Get script by script name

        Args:
            script_name: script name

        Returns:
            script matching script name

        """
        try:
            if script_name is None:
                return {}
            script_list = self._zapi.script.get({'output': 'extend', 'filter': {'name': [script_name]}})
            if len(script_list) < 1:
                self._module.fail_json(msg='Script not found: %s' % script_name)
            else:
                return script_list[0]
        except Exception as e:
            self._module.fail_json(msg="Failed to get script '%s': %s" % (script_name, e))