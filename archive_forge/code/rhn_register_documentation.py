from __future__ import absolute_import, division, print_function
import os
import sys
from ansible_collections.community.general.plugins.module_utils import redhat
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves import urllib, xmlrpc_client

            Return True if we are running against Hosted (rhn.redhat.com) or
            False otherwise (when running against Satellite or Spacewalk)
        