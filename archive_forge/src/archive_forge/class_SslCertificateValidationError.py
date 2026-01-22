from oslo_utils import encodeutils
from neutronclient._i18n import _
class SslCertificateValidationError(NeutronClientException):
    message = _('SSL certificate validation has failed: %(reason)s')