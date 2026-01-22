from __future__ import absolute_import, division, print_function
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
from ansible.module_utils._text import to_native
class NetAppESeriesLdap(NetAppESeriesModule):
    NO_CHANGE_MSG = 'No changes were necessary.'
    TEMPORARY_DOMAIN = 'ANSIBLE_TMP_DOMAIN'

    def __init__(self):
        ansible_options = dict(state=dict(type='str', required=False, default='present', choices=['present', 'absent', 'disabled']), identifier=dict(type='str', required=False, default='default'), bind_user=dict(type='str', required=False), bind_password=dict(type='str', required=False, no_log=True), names=dict(type='list', required=False), server_url=dict(type='str', required=False), search_base=dict(type='str', required=False), role_mappings=dict(type='dict', required=False, no_log=True), group_attributes=dict(type='list', default=['memberOf'], required=False), user_attribute=dict(type='str', required=False, default='sAMAccountName'))
        required_if = [['state', 'present', ['server_url']]]
        required_together = [['bind_user', 'bind_password']]
        super(NetAppESeriesLdap, self).__init__(ansible_options=ansible_options, web_services_version='02.00.0000.0000', required_if=required_if, required_together=required_together, supports_check_mode=True)
        args = self.module.params
        self.state = args['state']
        self.id = args['identifier']
        self.bind_user = args['bind_user']
        self.bind_password = args['bind_password']
        self.names = args['names']
        self.server = args['server_url']
        self.search_base = args['search_base']
        self.role_mappings = args['role_mappings']
        self.group_attributes = args['group_attributes']
        self.user_attribute = args['user_attribute']
        if self.server and (not self.names):
            parts = urlparse.urlparse(self.server)
            self.names = [parts.netloc.split(':')[0]]
        self.url_path_prefix = ''
        if self.is_embedded():
            self.url_path_prefix = 'storage-systems/1/'
        elif self.ssid != '0' and self.ssid.lower() != 'proxy':
            self.url_path_prefix = 'storage-systems/%s/forward/devmgr/v2/storage-systems/1/' % self.ssid
        self.existing_domain_ids = []
        self.domain = {}
        self.body = {}

    def get_domains(self):
        """Retrieve all domain information from storage system."""
        domains = None
        try:
            rc, response = self.request(self.url_path_prefix + 'ldap')
            domains = response['ldapDomains']
        except Exception as error:
            self.module.fail_json(msg='Failed to retrieve current LDAP configuration. Array Id [%s]. Error [%s].' % (self.ssid, to_native(error)))
        return domains

    def build_request_body(self):
        """Build the request body."""
        self.body.update({'id': self.id, 'groupAttributes': self.group_attributes, 'ldapUrl': self.server, 'names': self.names, 'roleMapCollection': []})
        if self.search_base:
            self.body.update({'searchBase': self.search_base})
        if self.user_attribute:
            self.body.update({'userAttribute': self.user_attribute})
        if self.bind_user and self.bind_password:
            self.body.update({'bindLookupUser': {'password': self.bind_password, 'user': self.bind_user}})
        if self.role_mappings:
            for regex, names in self.role_mappings.items():
                for name in names:
                    self.body['roleMapCollection'].append({'groupRegex': regex, 'ignorecase': True, 'name': name})

    def are_changes_required(self):
        """Determine whether any changes are required and build request body."""
        change_required = False
        domains = self.get_domains()
        if self.state == 'disabled' and domains:
            self.existing_domain_ids = [domain['id'] for domain in domains]
            change_required = True
        elif self.state == 'present':
            for domain in domains:
                if self.id == domain['id']:
                    self.domain = domain
                    if self.state == 'absent':
                        change_required = True
                    elif len(self.group_attributes) != len(domain['groupAttributes']) or any([a not in domain['groupAttributes'] for a in self.group_attributes]):
                        change_required = True
                    elif self.user_attribute != domain['userAttribute']:
                        change_required = True
                    elif self.search_base.lower() != domain['searchBase'].lower():
                        change_required = True
                    elif self.server != domain['ldapUrl']:
                        change_required = True
                    elif any((name not in domain['names'] for name in self.names)) or any((name not in self.names for name in domain['names'])):
                        change_required = True
                    elif self.role_mappings:
                        if len(self.body['roleMapCollection']) != len(domain['roleMapCollection']):
                            change_required = True
                        else:
                            for role_map in self.body['roleMapCollection']:
                                for existing_role_map in domain['roleMapCollection']:
                                    if role_map['groupRegex'] == existing_role_map['groupRegex'] and role_map['name'] == existing_role_map['name']:
                                        break
                                else:
                                    change_required = True
                    if not change_required and self.bind_user and self.bind_password:
                        if self.bind_user != domain['bindLookupUser']['user']:
                            change_required = True
                        elif self.bind_password:
                            temporary_domain = None
                            try:
                                if any((domain['id'] == self.TEMPORARY_DOMAIN for domain in domains)):
                                    self.delete_domain(self.TEMPORARY_DOMAIN)
                                temporary_domain = self.add_domain(temporary=True, skip_test=True)
                                rc, tests = self.request(self.url_path_prefix + 'ldap/test', method='POST')
                                temporary_domain_test = {}
                                domain_test = {}
                                for test in tests:
                                    if test['id'] == temporary_domain['id']:
                                        temporary_domain_test = test['result']
                                    if self.id == test['id']:
                                        domain_test = test['result']
                                if temporary_domain_test['authenticationTestResult'] == 'ok' and domain_test['authenticationTestResult'] != 'ok':
                                    change_required = True
                                elif temporary_domain_test['authenticationTestResult'] != 'ok':
                                    self.module.fail_json(msg='Failed to authenticate bind credentials! Array Id [%s].' % self.ssid)
                            finally:
                                if temporary_domain:
                                    self.delete_domain(self.TEMPORARY_DOMAIN)
                    break
            else:
                change_required = True
        elif self.state == 'absent':
            for domain in domains:
                if self.id == domain['id']:
                    change_required = True
        return change_required

    def add_domain(self, temporary=False, skip_test=False):
        """Add domain to storage system."""
        domain = None
        body = self.body.copy()
        if temporary:
            body.update({'id': self.TEMPORARY_DOMAIN, 'names': [self.TEMPORARY_DOMAIN]})
        try:
            rc, response = self.request(self.url_path_prefix + 'ldap/addDomain?skipTest=%s' % ('true' if not skip_test else 'false'), method='POST', data=body)
            domain = response['ldapDomains'][0]
        except Exception as error:
            self.module.fail_json(msg='Failed to create LDAP domain. Array Id [%s]. Error [%s].' % (self.ssid, to_native(error)))
        return domain

    def update_domain(self):
        """Update existing domain on storage system."""
        try:
            rc, response = self.request(self.url_path_prefix + 'ldap/%s' % self.domain['id'], method='POST', data=self.body)
        except Exception as error:
            self.module.fail_json(msg='Failed to update LDAP domain. Array Id [%s]. Error [%s].' % (self.ssid, to_native(error)))

    def delete_domain(self, domain_id):
        """Delete specific domain on the storage system."""
        try:
            url = self.url_path_prefix + 'ldap/%s' % domain_id
            rc, response = self.request(self.url_path_prefix + 'ldap/%s' % domain_id, method='DELETE')
        except Exception as error:
            self.module.fail_json(msg='Failed to delete LDAP domain. Array Id [%s]. Error [%s].' % (self.ssid, to_native(error)))

    def disable_domains(self):
        """Delete all existing domains on storage system."""
        for domain_id in self.existing_domain_ids:
            self.delete_domain(domain_id)

    def apply(self):
        """Apply any necessary changes to the LDAP configuration."""
        self.build_request_body()
        change_required = self.are_changes_required()
        if change_required and (not self.module.check_mode):
            if self.state == 'present':
                if self.domain:
                    self.update_domain()
                    self.module.exit_json(msg='LDAP domain has been updated. Array Id: [%s]' % self.ssid, changed=change_required)
                else:
                    self.add_domain()
                    self.module.exit_json(msg='LDAP domain has been added. Array Id: [%s]' % self.ssid, changed=change_required)
            elif self.state == 'absent':
                if self.domain:
                    self.delete_domain(self.domain['id'])
                    self.module.exit_json(msg='LDAP domain has been removed. Array Id: [%s]' % self.ssid, changed=change_required)
            else:
                self.disable_domains()
                self.module.exit_json(msg='All LDAP domains have been removed. Array Id: [%s]' % self.ssid, changed=change_required)
        self.module.exit_json(msg='No changes have been made to the LDAP configuration. Array Id: [%s]' % self.ssid, changed=change_required)