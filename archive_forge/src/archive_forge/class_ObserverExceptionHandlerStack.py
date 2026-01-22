import logging
import sys
class ObserverExceptionHandlerStack:
    """ A stack of exception handlers.

    Parameters
    ----------
    handlers : list of ObserverExceptionHandler
        The last item is the current handler.
    """

    def __init__(self):
        self.handlers = []

    def push_exception_handler(self, handler=None, reraise_exceptions=False):
        """ Push a new exception handler into the stack. Making it the
        current exception handler.

        Parameters
        ----------
        handler : callable(event) or None
            A callable to handle an event, in the context of
            an exception. If None, the exceptions will be logged.
        reraise_exceptions : boolean
            Whether to reraise the exception.
        """
        self.handlers.append(ObserverExceptionHandler(handler=handler, reraise_exceptions=reraise_exceptions))

    def pop_exception_handler(self):
        """ Pop the current exception handler from the stack.

        Raises
        ------
        IndexError
            If there are no handlers to pop.
        """
        return self.handlers.pop()

    def handle_exception(self, event):
        """ Handles a traits notification exception using the handler last pushed.

        Parameters
        ----------
        event : object
            An event object emitted by the notification.
        """
        _, excp, _ = sys.exc_info()
        try:
            handler_state = self.handlers[-1]
        except IndexError:
            handler_state = ObserverExceptionHandler(handler=None, reraise_exceptions=False)
        handler_state.handler(event)
        if handler_state.reraise_exceptions:
            raise excp