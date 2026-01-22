from __future__ import (absolute_import, division, print_function)
import os
import hashlib
import json
import socket
import struct
import traceback
import uuid
from functools import partial
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.module_utils.common.json import AnsibleJSONEncoder
from ansible.module_utils.six import iteritems
from ansible.module_utils.six.moves import cPickle
def __rpc__(self, name, *args, **kwargs):
    """Executes the json-rpc and returns the output received
           from remote device.
           :name: rpc method to be executed over connection plugin that implements jsonrpc 2.0
           :args: Ordered list of params passed as arguments to rpc method
           :kwargs: Dict of valid key, value pairs passed as arguments to rpc method

           For usage refer the respective connection plugin docs.
        """
    response = self._exec_jsonrpc(name, *args, **kwargs)
    if 'error' in response:
        err = response.get('error')
        msg = err.get('data') or err['message']
        code = err['code']
        raise ConnectionError(to_text(msg, errors='surrogate_then_replace'), code=code)
    return response['result']