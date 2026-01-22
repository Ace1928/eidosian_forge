from __future__ import (absolute_import, division, print_function)
import abc
import os
import json
import subprocess
from ansible.plugins.lookup import LookupBase
from ansible.errors import AnsibleLookupError, AnsibleOptionsError
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.module_utils.six import with_metaclass
from ansible_collections.community.general.plugins.module_utils.onepassword import OnePasswordConfig
class OnePassCLIv1(OnePassCLIBase):
    supports_version = '1'

    def _parse_field(self, data_json, field_name, section_title):
        """
        Retrieves the desired field from the `op` response payload

        When the item is a `password` type, the password is a key within the `details` key:

        $ op get item 'test item' | jq
        {
          [...]
          "templateUuid": "005",
          "details": {
            "notesPlain": "",
            "password": "foobar",
            "passwordHistory": [],
            "sections": [
              {
                "name": "linked items",
                "title": "Related Items"
              }
            ]
          },
          [...]
        }

        However, when the item is a `login` type, the password is within a fields array:

        $ op get item 'test item' | jq
        {
          [...]
          "details": {
            "fields": [
              {
                "designation": "username",
                "name": "username",
                "type": "T",
                "value": "foo"
              },
              {
                "designation": "password",
                "name": "password",
                "type": "P",
                "value": "bar"
              }
            ],
            [...]
          },
          [...]
        """
        data = json.loads(data_json)
        if section_title is None:
            if field_name in data['details']:
                return data['details'][field_name]
            for field_data in data['details'].get('fields', []):
                if field_data.get('name', '').lower() == field_name.lower():
                    return field_data.get('value', '')
        for section_data in data['details'].get('sections', []):
            if section_title is not None and section_title.lower() != section_data['title'].lower():
                continue
            for field_data in section_data.get('fields', []):
                if field_data.get('t', '').lower() == field_name.lower():
                    return field_data.get('v', '')
        return ''

    def assert_logged_in(self):
        args = ['get', 'account']
        if self.account_id:
            args.extend(['--account', self.account_id])
        elif self.subdomain:
            account = '{subdomain}.{domain}'.format(subdomain=self.subdomain, domain=self.domain)
            args.extend(['--account', account])
        rc, out, err = self._run(args, ignore_errors=True)
        return not bool(rc)

    def full_signin(self):
        if self.connect_host or self.connect_token:
            raise AnsibleLookupError('1Password Connect is not available with 1Password CLI version 1. Please use version 2 or later.')
        if self.service_account_token:
            raise AnsibleLookupError('1Password CLI version 1 does not support Service Accounts. Please use version 2 or later.')
        required_params = ['subdomain', 'username', 'secret_key', 'master_password']
        self._check_required_params(required_params)
        args = ['signin', '{0}.{1}'.format(self.subdomain, self.domain), to_bytes(self.username), to_bytes(self.secret_key), '--raw']
        return self._run(args, command_input=to_bytes(self.master_password))

    def get_raw(self, item_id, vault=None, token=None):
        args = ['get', 'item', item_id]
        if self.account_id:
            args.extend(['--account', self.account_id])
        if vault is not None:
            args += ['--vault={0}'.format(vault)]
        if token is not None:
            args += [to_bytes('--session=') + token]
        return self._run(args)

    def signin(self):
        self._check_required_params(['master_password'])
        args = ['signin', '--raw']
        if self.subdomain:
            args.append(self.subdomain)
        return self._run(args, command_input=to_bytes(self.master_password))