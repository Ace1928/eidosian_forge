from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def update_authentication(self, current_authentication, authentication_type, http_auth_enabled, http_login_form, http_strip_domains, http_case_sensitive, ldap_configured, ldap_auth_enabled, ldap_host, ldap_port, ldap_base_dn, ldap_search_attribute, ldap_bind_dn, ldap_case_sensitive, ldap_bind_password, ldap_userdirectory, saml_auth_enabled, saml_idp_entityid, saml_sso_url, saml_slo_url, saml_username_attribute, saml_sp_entityid, saml_nameid_format, saml_sign_messages, saml_sign_assertions, saml_sign_authn_requests, saml_sign_logout_requests, saml_sign_logout_responses, saml_encrypt_nameid, saml_encrypt_assertions, saml_case_sensitive, passwd_min_length, passwd_check_rules, ldap_jit_status, saml_jit_status, jit_provision_interval, disabled_usrgroup):
    try:
        params = {}
        if authentication_type:
            params['authentication_type'] = str(zabbix_utils.helper_to_numeric_value(['internal', 'ldap'], authentication_type))
        if isinstance(http_auth_enabled, bool):
            params['http_auth_enabled'] = str(int(http_auth_enabled))
        if http_login_form:
            params['http_login_form'] = str(zabbix_utils.helper_to_numeric_value(['zabbix_login_form', 'http_login_form'], http_login_form))
        if http_strip_domains:
            params['http_strip_domains'] = ','.join(http_strip_domains)
        if isinstance(http_case_sensitive, bool):
            params['http_case_sensitive'] = str(int(http_case_sensitive))
        if LooseVersion(self._zbx_api_version) < LooseVersion('6.4'):
            if isinstance(ldap_configured, bool):
                params['ldap_configured'] = str(int(ldap_configured))
        elif isinstance(ldap_auth_enabled, bool):
            params['ldap_auth_enabled'] = str(int(ldap_auth_enabled))
        if LooseVersion(self._zbx_api_version) < LooseVersion('6.2'):
            if ldap_host:
                params['ldap_host'] = ldap_host
            if ldap_port:
                params['ldap_port'] = str(ldap_port)
            if ldap_base_dn:
                params['ldap_base_dn'] = ldap_base_dn
            if ldap_search_attribute:
                params['ldap_search_attribute'] = ldap_search_attribute
            if ldap_bind_dn:
                params['ldap_bind_dn'] = ldap_bind_dn
            if ldap_bind_password:
                params['ldap_bind_password'] = ldap_bind_password
        elif ldap_userdirectory:
            directory = self._zapi.userdirectory.get({'search': {'name': ldap_userdirectory}})
            if not directory:
                self._module.fail_json(msg='Canot find user directory with name: %s' % ldap_userdirectory)
            params['ldap_userdirectoryid'] = directory[0]['userdirectoryid']
        if isinstance(ldap_case_sensitive, bool):
            params['ldap_case_sensitive'] = str(int(ldap_case_sensitive))
        if isinstance(saml_auth_enabled, bool):
            params['saml_auth_enabled'] = str(int(saml_auth_enabled))
        if LooseVersion(self._zbx_api_version) < LooseVersion('6.4'):
            if saml_idp_entityid:
                params['saml_idp_entityid'] = saml_idp_entityid
            if saml_sso_url:
                params['saml_sso_url'] = saml_sso_url
            if saml_slo_url:
                params['saml_slo_url'] = saml_slo_url
            if saml_username_attribute:
                params['saml_username_attribute'] = saml_username_attribute
            if saml_sp_entityid:
                params['saml_sp_entityid'] = saml_sp_entityid
            if saml_nameid_format:
                params['saml_nameid_format'] = saml_nameid_format
            if isinstance(saml_sign_messages, bool):
                params['saml_sign_messages'] = str(int(saml_sign_messages))
            if isinstance(saml_sign_assertions, bool):
                params['saml_sign_assertions'] = str(int(saml_sign_assertions))
            if isinstance(saml_sign_authn_requests, bool):
                params['saml_sign_authn_requests'] = str(int(saml_sign_authn_requests))
            if isinstance(saml_sign_logout_requests, bool):
                params['saml_sign_logout_requests'] = str(int(saml_sign_logout_requests))
            if isinstance(saml_sign_logout_responses, bool):
                params['saml_sign_logout_responses'] = str(int(saml_sign_logout_responses))
            if isinstance(saml_encrypt_nameid, bool):
                params['saml_encrypt_nameid'] = str(int(saml_encrypt_nameid))
            if isinstance(saml_encrypt_assertions, bool):
                params['saml_encrypt_assertions'] = str(int(saml_encrypt_assertions))
            if isinstance(saml_case_sensitive, bool):
                params['saml_case_sensitive'] = str(int(saml_case_sensitive))
        else:
            if isinstance(ldap_jit_status, bool):
                params['ldap_jit_status'] = str(int(ldap_jit_status))
            if isinstance(saml_jit_status, bool):
                params['saml_jit_status'] = str(int(saml_jit_status))
            if isinstance(jit_provision_interval, str):
                params['jit_provision_interval'] = jit_provision_interval
            if isinstance(disabled_usrgroup, str):
                usrgrpids = self._zapi.usergroup.get({'filter': {'name': disabled_usrgroup}})
                if not usrgrpids:
                    self._module.fail_json("User group '%s' cannot be found" % disabled_usrgroup)
                params['disabled_usrgrpid'] = str(int(usrgrpids[0]['usrgrpid']))
            if (ldap_jit_status or saml_jit_status) and (not disabled_usrgroup):
                self._module.fail_json("'disabled_usrgroup' must be provided if 'ldap_jit_status' or 'saml_jit_status' enabled")
        if passwd_min_length:
            if passwd_min_length < 1 or passwd_min_length > 70:
                self._module.fail_json(msg='Please set 0-70 to passwd_min_length.')
            else:
                params['passwd_min_length'] = str(passwd_min_length)
        if passwd_check_rules:
            passwd_check_rules_values = ['contain_uppercase_and_lowercase_letters', 'contain_digits', 'contain_special_characters', 'avoid_easy_to_guess']
            params['passwd_check_rules'] = 0
            if isinstance(passwd_check_rules, str):
                if passwd_check_rules not in passwd_check_rules_values:
                    self._module.fail_json(msg='%s is invalid value for passwd_check_rules.' % passwd_check_rules)
                params['passwd_check_rules'] += 2 ** zabbix_utils.helper_to_numeric_value(passwd_check_rules_values, passwd_check_rules)
            elif isinstance(passwd_check_rules, list):
                for _passwd_check_rules_value in passwd_check_rules:
                    if _passwd_check_rules_value not in passwd_check_rules_values:
                        self._module.fail_json(msg='%s is invalid value for passwd_check_rules.' % _passwd_check_rules_value)
                        params['passwd_check_rules'] += 2 ** zabbix_utils.helper_to_numeric_value(passwd_check_rules_values, _passwd_check_rules_value)
            params['passwd_check_rules'] = str(params['passwd_check_rules'])
        future_authentication = current_authentication.copy()
        future_authentication.update(params)
        if LooseVersion(self._zbx_api_version) < LooseVersion('6.4'):
            if current_authentication['ldap_configured'] == '0' and future_authentication['ldap_configured'] == '1':
                if LooseVersion(self._zbx_api_version) < LooseVersion('6.2'):
                    if not ldap_host or not ldap_port or (not ldap_search_attribute) or (not ldap_base_dn):
                        self._module.fail_json(msg='Please set ldap_host, ldap_search_attribute and ldap_base_dn when you change a value of ldap_configured to true.')
                elif not ldap_userdirectory:
                    self._module.fail_json(msg='Please set ldap_userdirectory when you change a value of ldap_configured to true.')
        elif current_authentication['ldap_auth_enabled'] == '0' and future_authentication['ldap_auth_enabled'] == '1':
            if not ldap_userdirectory:
                self._module.fail_json(msg='Please set ldap_userdirectory when you change a value of ldap_auth_enabled to true.')
        if LooseVersion(self._zbx_api_version) < LooseVersion('6.4'):
            if current_authentication['saml_auth_enabled'] == '0' and future_authentication['saml_auth_enabled'] == '1' and (not saml_idp_entityid) and (not saml_sso_url) and (not saml_username_attribute) and (not saml_sp_entityid):
                self._module.fail_json(msg=' '.join(['Please set saml_idp_entityid, saml_sso_url, saml_username_attribute and saml_sp_entityid', 'when you change a value of saml_auth_enabled to true.']))
        if future_authentication != current_authentication:
            if self._module.check_mode:
                self._module.exit_json(changed=True)
            self._zapi.authentication.update(params)
            self._module.exit_json(changed=True, result='Successfully update authentication setting')
        else:
            self._module.exit_json(changed=False, result='Authentication setting is already up to date')
    except Exception as e:
        self._module.fail_json(msg='Failed to update authentication setting, Exception: %s' % e)