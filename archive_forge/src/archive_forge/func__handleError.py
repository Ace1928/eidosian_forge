import fixtures
def _handleError(self, record):
    """Monkey patch for logging.Handler.handleError.

    The default handleError just logs the error to stderr but we want
    the option of actually raising an exception.
    """
    raise