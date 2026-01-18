from __future__ import absolute_import, division, print_function
import time
import socket
import re
import json
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import env_fallback
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list, EntityCollection
from ansible.module_utils.connection import Connection, exec_command
from ansible.module_utils.connection import ConnectionError
def waitForDeviceResponse(command, prompt, timeout, obj):
    obj.settimeout(int(timeout))
    obj.send(command)
    flag = False
    retVal = ''
    while not flag:
        time.sleep(1)
        try:
            buffByte = obj.recv(9999)
            buff = buffByte.decode()
            retVal = retVal + buff
            gotit = buff.find(prompt)
            if gotit != -1:
                flag = True
        except Exception:
            if prompt == '(yes/no)?':
                pass
            elif prompt == 'Password:':
                pass
            else:
                retVal = retVal + '\n Error-101'
            flag = True
    return retVal