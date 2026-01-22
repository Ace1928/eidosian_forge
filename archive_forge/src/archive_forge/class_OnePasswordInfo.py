from __future__ import (absolute_import, division, print_function)
import errno
import json
import os
import re
from subprocess import Popen, PIPE
from ansible.module_utils.common.text.converters import to_bytes, to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.onepassword import OnePasswordConfig
class OnePasswordInfo(object):

    def __init__(self):
        self.cli_path = module.params.get('cli_path')
        self.auto_login = module.params.get('auto_login')
        self.logged_in = False
        self.token = None
        terms = module.params.get('search_terms')
        self.terms = self.parse_search_terms(terms)
        self._config = OnePasswordConfig()

    def _run(self, args, expected_rc=0, command_input=None, ignore_errors=False):
        if self.token:
            args += [to_bytes('--session=') + self.token]
        command = [self.cli_path] + args
        p = Popen(command, stdout=PIPE, stderr=PIPE, stdin=PIPE)
        out, err = p.communicate(input=command_input)
        rc = p.wait()
        if not ignore_errors and rc != expected_rc:
            raise AnsibleModuleError(to_native(err))
        return (rc, out, err)

    def _parse_field(self, data_json, item_id, field_name, section_title=None):
        data = json.loads(data_json)
        if 'documentAttributes' in data['details']:
            document = self._run(['get', 'document', data['overview']['title']])
            return {'document': document[1].strip()}
        elif field_name in data['details']:
            return {field_name: data['details'][field_name]}
        else:
            if section_title is None:
                for field_data in data['details'].get('fields', []):
                    if field_data.get('name', '').lower() == field_name.lower():
                        return {field_name: field_data.get('value', '')}
            for section_data in data['details'].get('sections', []):
                if section_title is not None and section_title.lower() != section_data['title'].lower():
                    continue
                for field_data in section_data.get('fields', []):
                    if field_data.get('t', '').lower() == field_name.lower():
                        return {field_name: field_data.get('v', '')}
        optional_section_title = '' if section_title is None else " in the section '%s'" % section_title
        module.fail_json(msg="Unable to find an item in 1Password named '%s' with the field '%s'%s." % (item_id, field_name, optional_section_title))

    def parse_search_terms(self, terms):
        processed_terms = []
        for term in terms:
            if not isinstance(term, dict):
                term = {'name': term}
            if 'name' not in term:
                module.fail_json(msg="Missing required 'name' field from search term, got: '%s'" % to_native(term))
            term['field'] = term.get('field', 'password')
            term['section'] = term.get('section', None)
            term['vault'] = term.get('vault', None)
            processed_terms.append(term)
        return processed_terms

    def get_raw(self, item_id, vault=None):
        try:
            args = ['get', 'item', item_id]
            if vault is not None:
                args += ['--vault={0}'.format(vault)]
            rc, output, dummy = self._run(args)
            return output
        except Exception as e:
            if re.search('.*not found.*', to_native(e)):
                module.fail_json(msg="Unable to find an item in 1Password named '%s'." % item_id)
            else:
                module.fail_json(msg="Unexpected error attempting to find an item in 1Password named '%s': %s" % (item_id, to_native(e)))

    def get_field(self, item_id, field, section=None, vault=None):
        output = self.get_raw(item_id, vault)
        return self._parse_field(output, item_id, field, section) if output != '' else ''

    def full_login(self):
        if self.auto_login is not None:
            if None in [self.auto_login.get('subdomain'), self.auto_login.get('username'), self.auto_login.get('secret_key'), self.auto_login.get('master_password')]:
                module.fail_json(msg='Unable to perform initial sign in to 1Password. subdomain, username, secret_key, and master_password are required to perform initial sign in.')
            args = ['signin', '{0}.1password.com'.format(self.auto_login['subdomain']), to_bytes(self.auto_login['username']), to_bytes(self.auto_login['secret_key']), '--output=raw']
            try:
                rc, out, err = self._run(args, command_input=to_bytes(self.auto_login['master_password']))
                self.token = out.strip()
            except AnsibleModuleError as e:
                module.fail_json(msg='Failed to perform initial sign in to 1Password: %s' % to_native(e))
        else:
            module.fail_json(msg="Unable to perform an initial sign in to 1Password. Please run '%s signin' or define credentials in 'auto_login'. See the module documentation for details." % self.cli_path)

    def get_token(self):
        if os.path.isfile(self._config.config_file_path):
            if self.auto_login is not None:
                if not self.auto_login.get('master_password'):
                    module.fail_json(msg="Unable to sign in to 1Password. 'auto_login.master_password' is required.")
                try:
                    args = ['signin', '--output=raw']
                    if self.auto_login.get('subdomain'):
                        args = ['signin', self.auto_login['subdomain'], '--output=raw']
                    rc, out, err = self._run(args, command_input=to_bytes(self.auto_login['master_password']))
                    self.token = out.strip()
                except AnsibleModuleError:
                    self.full_login()
            else:
                self.full_login()
        else:
            self.full_login()

    def assert_logged_in(self):
        try:
            rc, out, err = self._run(['get', 'account'], ignore_errors=True)
            if rc == 0:
                self.logged_in = True
            if not self.logged_in:
                self.get_token()
        except OSError as e:
            if e.errno == errno.ENOENT:
                module.fail_json(msg="1Password CLI tool '%s' not installed in path on control machine" % self.cli_path)
            raise e

    def run(self):
        result = {}
        self.assert_logged_in()
        for term in self.terms:
            value = self.get_field(term['name'], term['field'], term['section'], term['vault'])
            if term['name'] in result:
                result[term['name']].update(value)
            else:
                result[term['name']] = value
        return result