import logging
import time
from urllib.parse import quote
class TransLogger:
    """
    This logging middleware will log all requests as they go through.
    They are, by default, sent to a logger named ``'wsgi'`` at the
    INFO level.

    If ``setup_console_handler`` is true, then messages for the named
    logger will be sent to the console.

    To adjust the format of the timestamp in the log, provide a strftime
    format string to the ``time_format`` keyword argument. Otherwise
    ``default_time_format`` will be used.
    """
    default_time_format = '%d/%b/%Y:%H:%M:%S '
    format = '%(REMOTE_ADDR)s - %(REMOTE_USER)s [%(time)s] "%(REQUEST_METHOD)s %(REQUEST_URI)s %(HTTP_VERSION)s" %(status)s %(bytes)s "%(HTTP_REFERER)s" "%(HTTP_USER_AGENT)s"'

    def __init__(self, application, logger=None, format=None, logging_level=logging.INFO, logger_name='wsgi', setup_console_handler=True, set_logger_level=logging.DEBUG, time_format=None):
        if format is not None:
            self.format = format
        self.time_format = time_format or self.default_time_format
        self.application = application
        self.logging_level = logging_level
        self.logger_name = logger_name
        if logger is None:
            self.logger = logging.getLogger(self.logger_name)
            if setup_console_handler:
                console = logging.StreamHandler()
                console.setLevel(logging.DEBUG)
                console.setFormatter(logging.Formatter('%(message)s'))
                self.logger.addHandler(console)
                self.logger.propagate = False
            if set_logger_level is not None:
                self.logger.setLevel(set_logger_level)
        else:
            self.logger = logger

    def __call__(self, environ, start_response):
        start = time.localtime()
        req_uri = quote(environ.get('SCRIPT_NAME', '') + environ.get('PATH_INFO', ''))
        if environ.get('QUERY_STRING'):
            req_uri += '?' + environ['QUERY_STRING']
        method = environ['REQUEST_METHOD']

        def replacement_start_response(status, headers, exc_info=None):
            bytes = None
            for name, value in headers:
                if name.lower() == 'content-length':
                    bytes = value
            self.write_log(environ, method, req_uri, start, status, bytes)
            return start_response(status, headers)
        return self.application(environ, replacement_start_response)

    def write_log(self, environ, method, req_uri, start, status, bytes):
        if bytes is None:
            bytes = '-'
        if time.daylight:
            offset = time.altzone / 60 / 60 * -100
        else:
            offset = time.timezone / 60 / 60 * -100
        if offset >= 0:
            offset = '+%0.4d' % offset
        elif offset < 0:
            offset = '%0.4d' % offset
        remote_addr = '-'
        if environ.get('HTTP_X_FORWARDED_FOR'):
            remote_addr = environ['HTTP_X_FORWARDED_FOR']
        elif environ.get('REMOTE_ADDR'):
            remote_addr = environ['REMOTE_ADDR']
        d = {'REMOTE_ADDR': remote_addr, 'REMOTE_USER': environ.get('REMOTE_USER') or '-', 'REQUEST_METHOD': method, 'REQUEST_URI': req_uri, 'HTTP_VERSION': environ.get('SERVER_PROTOCOL'), 'time': time.strftime(self.time_format, start) + offset, 'status': status.split(None, 1)[0], 'bytes': bytes, 'HTTP_REFERER': environ.get('HTTP_REFERER', '-'), 'HTTP_USER_AGENT': environ.get('HTTP_USER_AGENT', '-')}
        message = self.format % d
        self.logger.log(self.logging_level, message)