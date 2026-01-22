from __future__ import absolute_import, division, print_function
from uuid import uuid4
from ssl import CertificateError
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.connection import Connection
from ansible.module_utils._text import to_text
class ZabbixApiSection(object):
    parent = None
    name = None

    def __init__(self, parent, name):
        self.name = name
        self.parent = parent

    def __getattr__(self, name):

        def method(opts=None):
            if self.name == 'configuration' and name == 'import_':
                _method = 'configuration.import'
            else:
                _method = '%s.%s' % (self.name, name)
            if not opts:
                opts = {}
            payload = ZabbixApiRequest.payload_builder(_method, opts)
            return self.parent._httpapi_error_handle(payload=payload)
        return method