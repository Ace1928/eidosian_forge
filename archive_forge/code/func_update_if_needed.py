from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible.module_utils.urls import Request, SSLValidationError, ConnectionError
from ansible.module_utils.parsing.convert_bool import boolean as strtobool
from ansible.module_utils.six import PY2
from ansible.module_utils.six import raise_from, string_types
from ansible.module_utils.six.moves import StringIO
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.six.moves.http_cookiejar import CookieJar
from ansible.module_utils.six.moves.urllib.parse import urlparse, urlencode, quote
from ansible.module_utils.six.moves.configparser import ConfigParser, NoOptionError
from socket import getaddrinfo, IPPROTO_TCP
import time
import re
from json import loads, dumps
from os.path import isfile, expanduser, split, join, exists, isdir
from os import access, R_OK, getcwd, environ
def update_if_needed(self, existing_item, new_item, on_update=None, auto_exit=True, associations=None):
    response = None
    if existing_item:
        try:
            item_url = existing_item['url']
            item_type = existing_item['type']
            if item_type == 'user':
                item_name = existing_item['username']
            elif item_type == 'workflow_job_template_node':
                item_name = existing_item['identifier']
            elif item_type == 'credential_input_source':
                item_name = existing_item['id']
            elif item_type == 'instance':
                item_name = existing_item['hostname']
            else:
                item_name = existing_item['name']
            item_id = existing_item['id']
        except KeyError as ke:
            self.fail_json(msg='Unable to process update of item due to missing data {0}'.format(ke))
        needs_patch = self.objects_could_be_different(existing_item, new_item)
        self.json_output['id'] = item_id
        if needs_patch:
            response = self.patch_endpoint(item_url, **{'data': new_item})
            if response['status_code'] == 200:
                self.json_output['changed'] |= self.objects_could_be_different(existing_item, response['json'], field_set=new_item.keys(), warning=True)
            elif 'json' in response and '__all__' in response['json']:
                self.fail_json(msg=response['json']['__all__'])
            else:
                self.fail_json(**{'msg': 'Unable to update {0} {1}, see response'.format(item_type, item_name), 'response': response})
    else:
        raise RuntimeError('update_if_needed called incorrectly without existing_item')
    if associations is not None:
        for association_type, id_list in associations.items():
            endpoint = '{0}{1}/'.format(item_url, association_type)
            self.modify_associations(endpoint, id_list)
    if on_update is not None and self.json_output['changed']:
        if response is None:
            last_data = existing_item
        else:
            last_data = response['json']
        on_update(self, last_data)
    elif auto_exit:
        self.exit_json(**self.json_output)
    else:
        if response is None:
            last_data = existing_item
        else:
            last_data = response['json']
        return last_data