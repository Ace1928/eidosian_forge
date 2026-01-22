from __future__ import (absolute_import, division, print_function)
import hashlib
import json
import re
import uuid
import os
from collections import namedtuple
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable
from ansible.module_utils.six import iteritems
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMAuth
from ansible.errors import AnsibleParserError, AnsibleError
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils._text import to_native, to_bytes, to_text
from itertools import chain
def _credential_setup(self):
    auth_options = dict(auth_source=self.get_option('auth_source'), profile=self.get_option('profile'), subscription_id=self.get_option('subscription_id'), client_id=self.get_option('client_id'), secret=self.get_option('secret'), tenant=self.get_option('tenant'), ad_user=self.get_option('ad_user'), password=self.get_option('password'), cloud_environment=self.get_option('cloud_environment'), cert_validation_mode=self.get_option('cert_validation_mode'), api_profile=self.get_option('api_profile'), track1_cred=True, adfs_authority_url=self.get_option('adfs_authority_url'))
    self.azure_auth = AzureRMAuth(**auth_options)
    self._clientconfig = AzureRMRestConfiguration(self.azure_auth.azure_credential_track2, self.azure_auth.subscription_id, self.azure_auth._cloud_environment.endpoints.resource_manager)
    self.new_client = PipelineClient(self.azure_auth._cloud_environment.endpoints.resource_manager, config=self._clientconfig)