import hashlib
from base64 import b64encode
from libcloud.utils.py3 import ET, b, next, httplib
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import (
from libcloud.compute.providers import Provider
class OpenNebulaResponse(XmlResponse):
    """
    XmlResponse class for the OpenNebula.org driver.
    """

    def success(self):
        """
        Check if response has the appropriate HTTP response code to be a
        success.

        :rtype:  ``bool``
        :return: True is success, else False.
        """
        i = int(self.status)
        return 200 <= i <= 299

    def parse_error(self):
        """
        Check if response contains any errors.

        @raise: :class:`InvalidCredsError`

        :rtype:  :class:`ElementTree`
        :return: Contents of HTTP response body.
        """
        if int(self.status) == httplib.UNAUTHORIZED:
            raise InvalidCredsError(self.body)
        return self.body