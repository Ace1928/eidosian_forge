from pprint import pformat
from six import iteritems
import re
@tls.setter
def tls(self, tls):
    """
        Sets the tls of this NetworkingV1beta1IngressSpec.
        TLS configuration. Currently the Ingress only supports a single TLS
        port, 443. If multiple members of this list specify different hosts,
        they will be multiplexed on the same port according to the hostname
        specified through the SNI TLS extension, if the ingress controller
        fulfilling the ingress supports SNI.

        :param tls: The tls of this NetworkingV1beta1IngressSpec.
        :type: list[NetworkingV1beta1IngressTLS]
        """
    self._tls = tls