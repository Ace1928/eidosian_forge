from __future__ import absolute_import, division, print_function
import os
import re
import time
import uuid
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native

    Standard representation for a device as returned by various tasks::

        {
            'id': 'device_id'
            'hostname': 'device_hostname',
            'tags': [],
            'locked': false,
            'state': 'provisioning',
            'ip_addresses': [
                {
                    "address": "147.75.194.227",
                    "address_family": 4,
                    "public": true
                },
                {
                    "address": "2604:1380:2:5200::3",
                    "address_family": 6,
                    "public": true
                },
                {
                    "address": "10.100.11.129",
                    "address_family": 4,
                    "public": false
                }
            ],
            "private_ipv4": "10.100.11.129",
            "public_ipv4": "147.75.194.227",
            "public_ipv6": "2604:1380:2:5200::3",
        }

    