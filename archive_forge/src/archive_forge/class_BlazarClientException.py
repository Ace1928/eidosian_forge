from blazarclient.i18n import _
class BlazarClientException(Exception):
    """Base exception class."""
    message = _('An unknown exception occurred %s.')
    code = 500

    def __init__(self, message=None, **kwargs):
        self.kwargs = kwargs
        if 'code' not in self.kwargs:
            try:
                self.kwargs['code'] = self.code
            except AttributeError:
                pass
        if not message:
            message = self.message % kwargs
        super(BlazarClientException, self).__init__(message)