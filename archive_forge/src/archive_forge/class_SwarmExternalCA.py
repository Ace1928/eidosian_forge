from ..errors import InvalidVersion
from ..utils import version_lt
class SwarmExternalCA(dict):
    """
        Configuration for forwarding signing requests to an external
        certificate authority.

        Args:
            url (string): URL where certificate signing requests should be
                sent.
            protocol (string): Protocol for communication with the external CA.
            options (dict): An object with key/value pairs that are interpreted
                as protocol-specific options for the external CA driver.
            ca_cert (string): The root CA certificate (in PEM format) this
                external CA uses to issue TLS certificates (assumed to be to
                the current swarm root CA certificate if not provided).



    """

    def __init__(self, url, protocol=None, options=None, ca_cert=None):
        self['URL'] = url
        self['Protocol'] = protocol
        self['Options'] = options
        self['CACert'] = ca_cert