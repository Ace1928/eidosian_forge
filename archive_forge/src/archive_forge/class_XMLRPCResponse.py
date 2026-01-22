from typing import Dict, Type
from libcloud.utils.py3 import httplib, xmlrpclib
from libcloud.common.base import Response, Connection
from libcloud.common.types import LibcloudError
class XMLRPCResponse(ErrorCodeMixin, Response):
    defaultExceptionCls = Exception

    def success(self):
        return self.status == httplib.OK

    def parse_body(self):
        try:
            params, methodname = xmlrpclib.loads(self.body)
            if len(params) == 1:
                params = params[0]
            return params
        except xmlrpclib.Fault as e:
            self.raise_exception_for_error(e.faultCode, e.faultString)
            error_string = '{}: {}'.format(e.faultCode, e.faultString)
            raise self.defaultExceptionCls(error_string)

    def parse_error(self):
        msg = 'Server returned an invalid xmlrpc response (%d)' % self.status
        raise ProtocolError(msg)