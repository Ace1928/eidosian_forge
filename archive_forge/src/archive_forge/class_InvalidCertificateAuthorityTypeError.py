from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class InvalidCertificateAuthorityTypeError(exceptions.Error):
    """Error thrown for performing a command on the wrong CA type."""