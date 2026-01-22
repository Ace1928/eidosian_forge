import requests
import tempfile
from oslo_config import cfg
from oslo_config import sources
class URIConfigurationSource(sources.ConfigurationSource):
    """A configuration source for remote files served through http[s].

    :param uri: The Uniform Resource Identifier of the configuration to be
          retrieved.

    :param ca_path: The path to a CA_BUNDLE file or directory with
              certificates of trusted CAs.

    :param client_cert: Client side certificate, as a single file path
                  containing either the certificate only or the
                  private key and the certificate.

    :param client_key: Client side private key, in case client_cert is
                 specified but does not includes the private key.
    """

    def __init__(self, uri, ca_path=None, client_cert=None, client_key=None, timeout=60):
        self._uri = uri
        self._namespace = cfg._Namespace(cfg.ConfigOpts())
        data = self._fetch_uri(uri, ca_path, client_cert, client_key, timeout)
        with tempfile.NamedTemporaryFile() as tmpfile:
            tmpfile.write(data.encode('utf-8'))
            tmpfile.flush()
            cfg.ConfigParser._parse_file(tmpfile.name, self._namespace)

    def _fetch_uri(self, uri, ca_path, client_cert, client_key, timeout):
        verify = ca_path if ca_path else True
        cert = (client_cert, client_key) if client_cert and client_key else client_cert
        with requests.get(uri, verify=verify, cert=cert, timeout=timeout) as response:
            response.raise_for_status()
            return response.text

    def get(self, group_name, option_name, opt):
        try:
            return self._namespace._get_value([(group_name, option_name)], multi=opt.multi)
        except KeyError:
            return (sources._NoValue, None)