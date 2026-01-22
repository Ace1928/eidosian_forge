import sys
import traceback
import cgi
from io import StringIO
from paste.exceptions import formatter, collector, reporter
from paste import wsgilib
from paste import request
class ErrorMiddleware(object):
    """
    Error handling middleware

    Usage::

        error_catching_wsgi_app = ErrorMiddleware(wsgi_app)

    Settings:

      ``debug``:
          If true, then tracebacks will be shown in the browser.

      ``error_email``:
          an email address (or list of addresses) to send exception
          reports to

      ``error_log``:
          a filename to append tracebacks to

      ``show_exceptions_in_wsgi_errors``:
          If true, then errors will be printed to ``wsgi.errors``
          (frequently a server error log, or stderr).

      ``from_address``, ``smtp_server``, ``error_subject_prefix``, ``smtp_username``, ``smtp_password``, ``smtp_use_tls``:
          variables to control the emailed exception reports

      ``error_message``:
          When debug mode is off, the error message to show to users.

      ``xmlhttp_key``:
          When this key (default ``_``) is in the request GET variables
          (not POST!), expect that this is an XMLHttpRequest, and the
          response should be more minimal; it should not be a complete
          HTML page.

    Environment Configuration:

      ``paste.throw_errors``:
          If this setting in the request environment is true, then this
          middleware is disabled. This can be useful in a testing situation
          where you don't want errors to be caught and transformed.

      ``paste.expected_exceptions``:
          When this middleware encounters an exception listed in this
          environment variable and when the ``start_response`` has not
          yet occurred, the exception will be re-raised instead of being
          caught.  This should generally be set by middleware that may
          (but probably shouldn't be) installed above this middleware,
          and wants to get certain exceptions.  Exceptions raised after
          ``start_response`` have been called are always caught since
          by definition they are no longer expected.

    """

    def __init__(self, application, global_conf=None, debug=NoDefault, error_email=None, error_log=None, show_exceptions_in_wsgi_errors=NoDefault, from_address=None, smtp_server=None, smtp_username=None, smtp_password=None, smtp_use_tls=False, error_subject_prefix=None, error_message=None, xmlhttp_key=None):
        from paste.util import converters
        self.application = application
        if global_conf is None:
            global_conf = {}
        if debug is NoDefault:
            debug = converters.asbool(global_conf.get('debug'))
        if show_exceptions_in_wsgi_errors is NoDefault:
            show_exceptions_in_wsgi_errors = converters.asbool(global_conf.get('show_exceptions_in_wsgi_errors'))
        self.debug_mode = converters.asbool(debug)
        if error_email is None:
            error_email = global_conf.get('error_email') or global_conf.get('admin_email') or global_conf.get('webmaster_email') or global_conf.get('sysadmin_email')
        self.error_email = converters.aslist(error_email)
        self.error_log = error_log
        self.show_exceptions_in_wsgi_errors = show_exceptions_in_wsgi_errors
        if from_address is None:
            from_address = global_conf.get('error_from_address', 'errors@localhost')
        self.from_address = from_address
        if smtp_server is None:
            smtp_server = global_conf.get('smtp_server', 'localhost')
        self.smtp_server = smtp_server
        self.smtp_username = smtp_username or global_conf.get('smtp_username')
        self.smtp_password = smtp_password or global_conf.get('smtp_password')
        self.smtp_use_tls = smtp_use_tls or converters.asbool(global_conf.get('smtp_use_tls'))
        self.error_subject_prefix = error_subject_prefix or ''
        if error_message is None:
            error_message = global_conf.get('error_message')
        self.error_message = error_message
        if xmlhttp_key is None:
            xmlhttp_key = global_conf.get('xmlhttp_key', '_')
        self.xmlhttp_key = xmlhttp_key

    def __call__(self, environ, start_response):
        """
        The WSGI application interface.
        """
        if environ.get('paste.throw_errors'):
            return self.application(environ, start_response)
        environ['paste.throw_errors'] = True
        try:
            __traceback_supplement__ = (Supplement, self, environ)
            sr_checker = ResponseStartChecker(start_response)
            app_iter = self.application(environ, sr_checker)
            return self.make_catching_iter(app_iter, environ, sr_checker)
        except:
            exc_info = sys.exc_info()
            try:
                for expect in environ.get('paste.expected_exceptions', []):
                    if isinstance(exc_info[1], expect):
                        raise
                start_response('500 Internal Server Error', [('content-type', 'text/html')], exc_info)
                response = self.exception_handler(exc_info, environ)
                response = response.encode('utf8')
                return [response]
            finally:
                exc_info = None

    def make_catching_iter(self, app_iter, environ, sr_checker):
        if isinstance(app_iter, (list, tuple)):
            return app_iter
        return CatchingIter(app_iter, environ, sr_checker, self)

    def exception_handler(self, exc_info, environ):
        simple_html_error = False
        if self.xmlhttp_key:
            get_vars = request.parse_querystring(environ)
            if dict(get_vars).get(self.xmlhttp_key):
                simple_html_error = True
        return handle_exception(exc_info, environ['wsgi.errors'], html=True, debug_mode=self.debug_mode, error_email=self.error_email, error_log=self.error_log, show_exceptions_in_wsgi_errors=self.show_exceptions_in_wsgi_errors, error_email_from=self.from_address, smtp_server=self.smtp_server, smtp_username=self.smtp_username, smtp_password=self.smtp_password, smtp_use_tls=self.smtp_use_tls, error_subject_prefix=self.error_subject_prefix, error_message=self.error_message, simple_html_error=simple_html_error)