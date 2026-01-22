from oslo_vmware import service
class Vim(service.Service):
    """Service class that provides access to the VIM API."""

    def __init__(self, protocol='https', host='localhost', port=None, wsdl_url=None, cacert=None, insecure=True, pool_maxsize=10, connection_timeout=None, op_id_prefix='oslo.vmware'):
        """Constructs a VIM service client object.

        :param protocol: http or https
        :param host: server IP address or host name
        :param port: port for connection
        :param wsdl_url: VIM WSDL url
        :param cacert: Specify a CA bundle file to use in verifying a
                       TLS (https) server certificate.
        :param insecure: Verify HTTPS connections using system certificates,
                         used only if cacert is not specified
        :param pool_maxsize: Maximum number of connections in http
                             connection pool
        :param connection_timeout: Maximum time in seconds to wait for peer to
                                   respond.
        :param op_id_prefix: String prefix for the operation ID.
        :raises: VimException, VimFaultException, VimAttributeException,
                 VimSessionOverLoadException, VimConnectionException
        """
        base_url = service.Service.build_base_url(protocol, host, port)
        soap_url = base_url + '/sdk'
        if wsdl_url is None:
            wsdl_url = soap_url + '/vimService.wsdl'
        super(Vim, self).__init__(wsdl_url, soap_url, cacert, insecure, pool_maxsize, connection_timeout, op_id_prefix)

    def retrieve_service_content(self):
        return self.RetrieveServiceContent(service.SERVICE_INSTANCE)

    def __repr__(self):
        return 'VIM Object'

    def __str__(self):
        return 'VIM Object'