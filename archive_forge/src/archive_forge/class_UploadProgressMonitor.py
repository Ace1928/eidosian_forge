import time
from paste.wsgilib import catch_errors
class UploadProgressMonitor(object):
    """
    monitors and reports on the status of uploads in progress

    Parameters:

        ``application``

            This is the next application in the WSGI stack.

        ``threshold``

            This is the size in bytes that is needed for the
            upload to be included in the monitor.

        ``timeout``

            This is the amount of time (in seconds) that a upload
            remains in the monitor after it has finished.

    Methods:

        ``uploads()``

            This returns a list of ``environ`` dict objects for each
            upload being currently monitored, or finished but whose time
            has not yet expired.

    For each request ``environ`` that is monitored, there are several
    variables that are stored:

        ``paste.bytes_received``

            This is the total number of bytes received for the given
            request; it can be compared with ``CONTENT_LENGTH`` to
            build a percentage complete.  This is an integer value.

        ``paste.request_started``

            This is the time (in seconds) when the request was started
            as obtained from ``time.time()``.  One would want to format
            this for presentation to the user, if necessary.

        ``paste.request_finished``

            This is the time (in seconds) when the request was finished,
            canceled, or otherwise disconnected.  This is None while
            the given upload is still in-progress.

    TODO: turn monitor into a queue and purge queue of finished
          requests that have passed the timeout period.
    """

    def __init__(self, application, threshold=None, timeout=None):
        self.application = application
        self.threshold = threshold or DEFAULT_THRESHOLD
        self.timeout = timeout or DEFAULT_TIMEOUT
        self.monitor = []

    def __call__(self, environ, start_response):
        length = environ.get('CONTENT_LENGTH', 0)
        if length and int(length) > self.threshold:
            self.monitor.append(environ)
            environ[ENVIRON_RECEIVED] = 0
            environ[REQUEST_STARTED] = time.time()
            environ[REQUEST_FINISHED] = None
            environ['wsgi.input'] = _ProgressFile(environ, environ['wsgi.input'])

            def finalizer(exc_info=None):
                environ[REQUEST_FINISHED] = time.time()
            return catch_errors(self.application, environ, start_response, finalizer, finalizer)
        return self.application(environ, start_response)

    def uploads(self):
        return self.monitor