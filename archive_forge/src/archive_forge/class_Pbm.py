import logging
import os
import urllib.parse as urlparse
import urllib.request as urllib
from oslo_vmware import service
from oslo_vmware import vim_util
class Pbm(service.Service):
    """Service class that provides access to the Storage Policy API."""

    def __init__(self, protocol='https', host='localhost', port=443, wsdl_url=None, cacert=None, insecure=True, pool_maxsize=10, connection_timeout=None, op_id_prefix='oslo.vmware'):
        """Constructs a PBM service client object.

        :param protocol: http or https
        :param host: server IP address or host name
        :param port: port for connection
        :param wsdl_url: PBM WSDL url
        :param cacert: Specify a CA bundle file to use in verifying a
                       TLS (https) server certificate.
        :param insecure: Verify HTTPS connections using system certificates,
                         used only if cacert is not specified
        :param pool_maxsize: Maximum number of connections in http
                             connection pool
        :param op_id_prefix: String prefix for the operation ID.
        :param connection_timeout: Maximum time in seconds to wait for peer to
                                   respond.
        """
        base_url = service.Service.build_base_url(protocol, host, port)
        soap_url = base_url + '/pbm'
        super(Pbm, self).__init__(wsdl_url, soap_url, cacert, insecure, pool_maxsize, connection_timeout, op_id_prefix)

    def set_soap_cookie(self, cookie):
        """Set the specified vCenter session cookie in the SOAP header

        :param cookie: cookie to set
        """
        self._vc_session_cookie = cookie

    def retrieve_service_content(self):
        ref = vim_util.get_moref(service.SERVICE_INSTANCE, SERVICE_TYPE)
        return self.PbmRetrieveServiceContent(ref)

    def __repr__(self):
        return 'PBM Object'

    def __str__(self):
        return 'PBM Object'