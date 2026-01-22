class IntegrationException(Exception):
    """Base Tempest Exception.

    To correctly use this class, inherit from it and define
    a 'message' property. That message will get printf'd
    with the keyword arguments provided to the constructor.
    """
    message = 'An unknown exception occurred'

    def __init__(self, *args, **kwargs):
        super(IntegrationException, self).__init__()
        try:
            self._error_string = self.message % kwargs
        except Exception:
            self._error_string = self.message
        if len(args) > 0:
            args = ['%s' % arg for arg in args]
            self._error_string = self._error_string + '\nDetails: %s' % '\n'.join(args)

    def __str__(self):
        return self._error_string