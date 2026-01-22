from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import itertools
import re
from googlecloudsdk.calliope import arg_parsers
class PemCertificatesFile(arg_parsers.FileContents):
    """Reads file from provided path, extracts all PEM certificates from that file, and packs them into a message format appropriate for use in the Trust Store."""
    PEM_RE = re.compile('-----BEGIN CERTIFICATE-----\\s*[\\r\\n|\\r|\\n][\\w\\s+/=]+[\\r\\n|\\r|\\n]-----END CERTIFICATE-----', re.DOTALL | re.ASCII)

    def __init__(self):
        super(PemCertificatesFile, self).__init__(binary=False)

    def __call__(self, name):
        file_contents = super(PemCertificatesFile, self).__call__(name)
        certs = self.PEM_RE.findall(file_contents)
        return [{'pemCertificate': cert} for cert in certs]