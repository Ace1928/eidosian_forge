from pprint import pformat
from six import iteritems
import re
@usages.setter
def usages(self, usages):
    """
        Sets the usages of this V1beta1CertificateSigningRequestSpec.
        allowedUsages specifies a set of usage contexts the key will be valid
        for. See: https://tools.ietf.org/html/rfc5280#section-4.2.1.3
        https://tools.ietf.org/html/rfc5280#section-4.2.1.12

        :param usages: The usages of this V1beta1CertificateSigningRequestSpec.
        :type: list[str]
        """
    self._usages = usages