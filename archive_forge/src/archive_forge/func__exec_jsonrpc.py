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
def _exec_jsonrpc(self, name, *args, **kwargs):
    req = request_builder(name, *args, **kwargs)
    reqid = req['id']
    if not os.path.exists(self.socket_path):
        raise ConnectionError('socket path %s does not exist or cannot be found. See Troubleshooting socket path issues in the Network Debug and Troubleshooting Guide' % self.socket_path)
    try:
        data = json.dumps(req, cls=AnsibleJSONEncoder, vault_to_text=True)
    except TypeError as exc:
        raise ConnectionError('Failed to encode some variables as JSON for communication with ansible-connection. The original exception was: %s' % to_text(exc))
    try:
        out = self.send(data)
    except socket.error as e:
        raise ConnectionError('unable to connect to socket %s. See Troubleshooting socket path issues in the Network Debug and Troubleshooting Guide' % self.socket_path, err=to_text(e, errors='surrogate_then_replace'), exception=traceback.format_exc())
    try:
        response = json.loads(out)
    except ValueError:
        if name.startswith('set_option'):
            raise ConnectionError("Unable to decode JSON from response to {0}. Received '{1}'.".format(name, out))
        params = [repr(arg) for arg in args] + ['{0}={1!r}'.format(k, v) for k, v in iteritems(kwargs)]
        params = ', '.join(params)
        raise ConnectionError("Unable to decode JSON from response to {0}({1}). Received '{2}'.".format(name, params, out))
    if response['id'] != reqid:
        raise ConnectionError('invalid json-rpc id received')
    if 'result_type' in response:
        response['result'] = cPickle.loads(to_bytes(response['result']))
    return response