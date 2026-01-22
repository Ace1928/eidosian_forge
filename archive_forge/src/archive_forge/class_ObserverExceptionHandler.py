import logging
import sys
class ObserverExceptionHandler:
    """ State for an exception handler.

    Parameters
    ----------
    handler : callable(event) or None
        A callable to handle an event, in the context of
        an exception. If None, the exceptions will be logged.
    reraise_exceptions : boolean
        Whether to reraise the exception.
    """

    def __init__(self, handler, reraise_exceptions):
        self.handler = handler if handler is not None else self._log_exception
        self.reraise_exceptions = reraise_exceptions

    def _log_exception(self, event):
        """ A handler that logs the exception with the given event.

        Parameters
        ----------
        event : object
            An event object emitted by the notification.
        """
        _logger.exception('Exception occurred in traits notification handler for event object: %r', event)