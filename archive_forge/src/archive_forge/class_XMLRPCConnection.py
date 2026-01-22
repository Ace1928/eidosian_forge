from typing import Dict, Type
from libcloud.utils.py3 import httplib, xmlrpclib
from libcloud.common.base import Response, Connection
from libcloud.common.types import LibcloudError
class XMLRPCConnection(Connection):
    """
    Connection class which can call XMLRPC based API's.

    This class uses the xmlrpclib marshalling and demarshalling code but uses
    the http transports provided by libcloud giving it better certificate
    validation and debugging helpers than the core client library.
    """
    responseCls = XMLRPCResponse
    endpoint = None

    def add_default_headers(self, headers):
        headers['Content-Type'] = 'text/xml'
        return headers

    def request(self, method_name, *args, **kwargs):
        """
        Call a given `method_name`.

        :type method_name: ``str``
        :param method_name: A method exposed by the xmlrpc endpoint that you
            are connecting to.

        :type args: ``tuple``
        :param args: Arguments to invoke with method with.
        """
        endpoint = kwargs.get('endpoint', self.endpoint)
        data = xmlrpclib.dumps(args, methodname=method_name, allow_none=True)
        return super().request(endpoint, data=data, method='POST')