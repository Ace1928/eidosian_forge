from __future__ import absolute_import, division, print_function
import os
import sys
import copy
import json
import logging
import time
from datetime import datetime, timedelta
from ssl import SSLError
def update_from_ansible_module(self, m):
    """
        :param m: ansible module
        :return:
        """
    if m.params.get('avi_credentials'):
        for k, v in m.params['avi_credentials'].items():
            if hasattr(self, k):
                setattr(self, k, v)
    if m.params['controller']:
        self.controller = m.params['controller']
    if m.params['username']:
        self.username = m.params['username']
    if m.params['password']:
        self.password = m.params['password']
    if m.params['api_version'] and m.params['api_version'] != '16.4.4':
        self.api_version = m.params['api_version']
    if m.params['tenant']:
        self.tenant = m.params['tenant']
    if m.params['tenant_uuid']:
        self.tenant_uuid = m.params['tenant_uuid']
    if m.params.get('session_id'):
        self.session_id = m.params['session_id']
    if m.params.get('csrftoken'):
        self.csrftoken = m.params['csrftoken']