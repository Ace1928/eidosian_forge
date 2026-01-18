from __future__ import absolute_import, division, print_function
import socket
import struct
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
 Send a magic Wake-on-LAN packet. 