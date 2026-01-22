import os
from base64 import b64encode
from collections import deque
from http.client import HTTPConnection
from json import loads
from threading import Event, Thread
from time import sleep
from urllib.parse import urlparse, urlunparse
import requests
from kivy.clock import Clock
from kivy.config import Config
from kivy.logger import Logger
from kivy.utils import platform
from kivy.weakmethod import WeakMethod
class UrlRequestBase(Thread):
    """A UrlRequest. See module documentation for usage.

    .. versionchanged:: 1.5.1
        Add `debug` parameter

    .. versionchanged:: 1.0.10
        Add `method` parameter

    .. versionchanged:: 1.8.0

        Parameter `decode` added.
        Parameter `file_path` added.
        Parameter `on_redirect` added.
        Parameter `on_failure` added.

    .. versionchanged:: 1.9.1

        Parameter `ca_file` added.
        Parameter `verify` added.

    .. versionchanged:: 1.10.0

        Parameters `proxy_host`, `proxy_port` and `proxy_headers` added.

    .. versionchanged:: 1.11.0

        Parameters `on_cancel` added.

    .. versionchanged:: 2.2.0

        Parameters `on_finish` added.
        Parameters `auth` added.

    :Parameters:
        `url`: str
            Complete url string to call.
        `on_success`: callback(request, result)
            Callback function to call when the result has been fetched.
        `on_redirect`: callback(request, result)
            Callback function to call if the server returns a Redirect.
        `on_failure`: callback(request, result)
            Callback function to call if the server returns a Client or
            Server Error.
        `on_error`: callback(request, error)
            Callback function to call if an error occurs.
        `on_progress`: callback(request, current_size, total_size)
            Callback function that will be called to report progression of the
            download. `total_size` might be -1 if no Content-Length has been
            reported in the http response.
            This callback will be called after each `chunk_size` is read.
        `on_cancel`: callback(request)
            Callback function to call if user requested to cancel the download
            operation via the .cancel() method.
        `on_finish`: callback(request)
            Additional callback function to call if request is done.
        `req_body`: str, defaults to None
            Data to sent in the request. If it's not None, a POST will be done
            instead of a GET.
        `req_headers`: dict, defaults to None
            Custom headers to add to the request.
        `chunk_size`: int, defaults to 8192
            Size of each chunk to read, used only when `on_progress` callback
            has been set. If you decrease it too much, a lot of on_progress
            callbacks will be fired and will slow down your download. If you
            want to have the maximum download speed, increase the chunk_size
            or don't use ``on_progress``.
        `timeout`: int, defaults to None
            If set, blocking operations will timeout after this many seconds.
        `method`: str, defaults to 'GET' (or 'POST' if ``body`` is specified)
            The HTTP method to use.
        `decode`: bool, defaults to True
            If False, skip decoding of the response.
        `debug`: bool, defaults to False
            If True, it will use the Logger.debug to print information
            about url access/progression/errors.
        `file_path`: str, defaults to None
            If set, the result of the UrlRequest will be written to this path
            instead of in memory.
        `ca_file`: str, defaults to None
            Indicates a SSL CA certificate file path to validate HTTPS
            certificates against
        `verify`: bool, defaults to True
            If False, disables SSL CA certificate verification
        `proxy_host`: str, defaults to None
            If set, the proxy host to use for this connection.
        `proxy_port`: int, defaults to None
            If set, and `proxy_host` is also set, the port to use for
            connecting to the proxy server.
        `proxy_headers`: dict, defaults to None
            If set, and `proxy_host` is also set, the headers to send to the
            proxy server in the ``CONNECT`` request.
        `auth`: HTTPBasicAuth, defaults to None
            If set, request will use basicauth to authenticate.
            Only used in "Requests" implementation
    """

    def __init__(self, url, on_success=None, on_redirect=None, on_failure=None, on_error=None, on_progress=None, req_body=None, req_headers=None, chunk_size=8192, timeout=None, method=None, decode=True, debug=False, file_path=None, ca_file=None, verify=True, proxy_host=None, proxy_port=None, proxy_headers=None, user_agent=None, on_cancel=None, on_finish=None, cookies=None, auth=None):
        super().__init__()
        self._queue = deque()
        self._trigger_result = Clock.create_trigger(self._dispatch_result, 0)
        self.daemon = True
        self.on_success = WeakMethod(on_success) if on_success else None
        self.on_redirect = WeakMethod(on_redirect) if on_redirect else None
        self.on_failure = WeakMethod(on_failure) if on_failure else None
        self.on_error = WeakMethod(on_error) if on_error else None
        self.on_progress = WeakMethod(on_progress) if on_progress else None
        self.on_cancel = WeakMethod(on_cancel) if on_cancel else None
        self.on_finish = WeakMethod(on_finish) if on_finish else None
        self.decode = decode
        self.file_path = file_path
        self._debug = debug
        self._result = None
        self._error = None
        self._is_finished = False
        self._resp_status = None
        self._resp_headers = None
        self._resp_length = -1
        self._chunk_size = chunk_size
        self._timeout = timeout
        self._method = method
        self.verify = verify
        self._proxy_host = proxy_host
        self._proxy_port = proxy_port
        self._proxy_headers = proxy_headers
        self._cancel_event = Event()
        self._user_agent = user_agent
        self._cookies = cookies
        self._requested_url = url
        self._auth = auth
        if platform in ['android', 'ios']:
            import certifi
            self.ca_file = ca_file or certifi.where()
        else:
            self.ca_file = ca_file
        self.url = url
        self.req_body = req_body
        self.req_headers = req_headers
        g_requests.append(self)
        self.start()

    def run(self):
        q = self._queue.appendleft
        url = self.url
        req_body = self.req_body
        req_headers = self.req_headers or {}
        user_agent = self._user_agent
        cookies = self._cookies
        if user_agent:
            req_headers.setdefault('User-Agent', user_agent)
        elif Config.has_section('network') and 'useragent' in Config.items('network'):
            useragent = Config.get('network', 'useragent')
            req_headers.setdefault('User-Agent', useragent)
        if cookies:
            req_headers.setdefault('Cookie', cookies)
        try:
            result, resp = self._fetch_url(url, req_body, req_headers, q)
            if self.decode:
                result = self.decode_result(result, resp)
        except Exception as e:
            q(('error', None, e))
        else:
            if not self._cancel_event.is_set():
                q(('success', resp, result))
            else:
                q(('killed', None, None))
        self._trigger_result()
        while len(self._queue):
            sleep(0.1)
            self._trigger_result()
        if self in g_requests:
            g_requests.remove(self)

    def _fetch_url(self, url, body, headers, q):
        trigger = self._trigger_result
        chunk_size = self._chunk_size
        report_progress = self.on_progress is not None
        file_path = self.file_path
        if self._debug:
            Logger.debug('UrlRequest: {0} Fetch url <{1}>'.format(id(self), url))
            Logger.debug('UrlRequest: {0} - body: {1}'.format(id(self), body))
            Logger.debug('UrlRequest: {0} - headers: {1}'.format(id(self), headers))
        req, resp = self.call_request(body, headers)
        if report_progress or file_path is not None:
            total_size = self.get_total_size(resp)
            if report_progress:
                q(('progress', resp, (0, total_size)))
            if file_path is not None:
                with open(file_path, 'wb') as fd:
                    bytes_so_far, result = self.get_chunks(resp, chunk_size, total_size, report_progress, q, trigger, fd=fd)
            else:
                bytes_so_far, result = self.get_chunks(resp, chunk_size, total_size, report_progress, q, trigger)
            if report_progress:
                q(('progress', resp, (bytes_so_far, total_size)))
                trigger()
        else:
            result = self.get_response(resp)
            try:
                if isinstance(result, bytes):
                    result = result.decode('utf-8')
            except UnicodeDecodeError:
                pass
        self.close_connection(req)
        return (result, resp)

    def decode_result(self, result, resp):
        """Decode the result fetched from url according to his Content-Type.
        Currently supports only application/json.
        """
        content_type = self.get_content_type(resp)
        if content_type is not None:
            ct = content_type.split(';')[0]
            if ct == 'application/json':
                if isinstance(result, bytes):
                    result = result.decode('utf-8')
                try:
                    return loads(result)
                except Exception:
                    return result
        return result

    def _dispatch_result(self, dt):
        while True:
            try:
                result, resp, data = self._queue.pop()
            except IndexError:
                return
            if resp:
                final_cookies = ''
                parsed_headers = []
                for key, value in self.get_all_headers(resp):
                    if key == 'Set-Cookie':
                        final_cookies += '{};'.format(value)
                    else:
                        parsed_headers.append((key, value))
                parsed_headers.append(('Set-Cookie', final_cookies[:-1]))
                self._resp_headers = dict(parsed_headers)
                self._resp_status = self.get_status_code(resp)
            if result == 'success':
                status_class = self.get_status_code(resp) // 100
                if status_class in (1, 2):
                    if self._debug:
                        Logger.debug('UrlRequest: {0} Download finished with {1} datalen'.format(id(self), data))
                    self._is_finished = True
                    self._result = data
                    if self.on_success:
                        func = self.on_success()
                        if func:
                            func(self, data)
                elif status_class == 3:
                    if self._debug:
                        Logger.debug('UrlRequest: {} Download redirected'.format(id(self)))
                    self._is_finished = True
                    self._result = data
                    if self.on_redirect:
                        func = self.on_redirect()
                        if func:
                            func(self, data)
                elif status_class in (4, 5):
                    if self._debug:
                        Logger.debug('UrlRequest: {} Download failed with http error {}'.format(id(self), self.get_status_code(resp)))
                    self._is_finished = True
                    self._result = data
                    if self.on_failure:
                        func = self.on_failure()
                        if func:
                            func(self, data)
            elif result == 'error':
                if self._debug:
                    Logger.debug('UrlRequest: {0} Download error <{1}>'.format(id(self), data))
                self._is_finished = True
                self._error = data
                if self.on_error:
                    func = self.on_error()
                    if func:
                        func(self, data)
            elif result == 'progress':
                if self._debug:
                    Logger.debug('UrlRequest: {0} Download progress {1}'.format(id(self), data))
                if self.on_progress:
                    func = self.on_progress()
                    if func:
                        func(self, data[0], data[1])
            elif result == 'killed':
                if self._debug:
                    Logger.debug('UrlRequest: Cancelled by user')
                if self.on_cancel:
                    func = self.on_cancel()
                    if func:
                        func(self)
            else:
                assert 0
            if result != 'progress' and self.on_finish:
                if self._debug:
                    Logger.debug('UrlRequest: Request is finished')
                func = self.on_finish()
                if func:
                    func(self)

    @property
    def is_finished(self):
        """Return True if the request has finished, whether it's a
        success or a failure.
        """
        return self._is_finished

    @property
    def result(self):
        """Return the result of the request.
        This value is not determined until the request is finished.
        """
        return self._result

    @property
    def resp_headers(self):
        """If the request has been completed, return a dictionary containing
        the headers of the response. Otherwise, it will return None.
        """
        return self._resp_headers

    @property
    def resp_status(self):
        """Return the status code of the response if the request is complete,
        otherwise return None.
        """
        return self._resp_status

    @property
    def error(self):
        """Return the error of the request.
        This value is not determined until the request is completed.
        """
        return self._error

    @property
    def chunk_size(self):
        """Return the size of a chunk, used only in "progress" mode (when
        on_progress callback is set.)
        """
        return self._chunk_size

    def wait(self, delay=0.5):
        """Wait for the request to finish (until :attr:`resp_status` is not
        None)

        .. note::
            This method is intended to be used in the main thread, and the
            callback will be dispatched from the same thread
            from which you're calling.

        .. versionadded:: 1.1.0
        """
        while self.resp_status is None:
            self._dispatch_result(delay)
            sleep(delay)

    def cancel(self):
        """Cancel the current request. It will be aborted, and the result
        will not be dispatched. Once cancelled, the callback on_cancel will
        be called.

        .. versionadded:: 1.11.0
        """
        self._cancel_event.set()