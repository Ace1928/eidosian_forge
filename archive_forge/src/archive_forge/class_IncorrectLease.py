from blazarclient.i18n import _
class IncorrectLease(BlazarClientException):
    """Occurs if lease parameters are incorrect."""
    message = _('The lease parameters are incorrect.')
    code = 409