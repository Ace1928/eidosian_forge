import sys
import time
import datetime
import itertools
from libcloud.pricing import get_pricing
from libcloud.common.base import LazyObject
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
class GCESslCertificate(UuidMixin):
    """GCESslCertificate represents the SslCertificate resource."""

    def __init__(self, id, name, certificate, driver, extra, private_key=None, description=None):
        """
        :param  name:  Name of the resource. Provided by the client when the
                       resource is created. The name must be 1-63 characters
                       long, and comply with RFC1035. Specifically, the name
                       must be 1-63 characters long and match the regular
                       expression [a-z]([-a-z0-9]*[a-z0-9])? which means the
                       first character must be a lowercase letter, and all
                       following characters must be a dash, lowercase letter,
                       or digit, except the last character, which cannot be a
                       dash.
        :type   name: ``str``

        :param  certificate:  A local certificate file. The certificate must
                              be in PEM format. The certificate chain must be
                              no greater than 5 certs long. The chain must
                              include at least one intermediate cert.
        :type   certificate: ``str``

        :param  private_key:  A write-only private key in PEM format. Only
                              insert RPCs will include this field.
        :type   private_key: ``str``

        :keyword  description:  An optional description of this resource.
                              Provide this property when you create the
                              resource.
        :type   description: ``str``

        :keyword  driver:  An initialized :class: `GCENodeDriver`
        :type   driver: :class:`:class: `GCENodeDriver``

        :keyword  extra:  A dictionary of extra information.
        :type   extra: ``:class: ``dict````

        """
        self.name = name
        self.certificate = certificate
        self.private_key = private_key
        self.description = description
        self.driver = driver
        self.extra = extra
        UuidMixin.__init__(self)

    def __repr__(self):
        return '<GCESslCertificate name="%s">' % self.name

    def destroy(self):
        """
        Destroy this SslCertificate.

        :return:  Return True if successful.
        :rtype: ``bool``
        """
        return self.driver.ex_destroy_sslcertificate(sslcertificate=self)