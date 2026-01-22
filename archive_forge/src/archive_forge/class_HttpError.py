class HttpError(CommunicationError):
    """Error making a request. Soon to be HttpError."""

    def __init__(self, response, content, url, method_config=None, request=None):
        error_message = HttpError._build_message(response, content, url)
        super(HttpError, self).__init__(error_message)
        self.response = response
        self.content = content
        self.url = url
        self.method_config = method_config
        self.request = request

    def __str__(self):
        return HttpError._build_message(self.response, self.content, self.url)

    @staticmethod
    def _build_message(response, content, url):
        if isinstance(content, bytes):
            content = content.decode('ascii', 'replace')
        return 'HttpError accessing <%s>: response: <%s>, content <%s>' % (url, response, content)

    @property
    def status_code(self):
        return int(self.response['status'])

    @classmethod
    def FromResponse(cls, http_response, **kwargs):
        try:
            status_code = int(http_response.info.get('status'))
            error_cls = _HTTP_ERRORS.get(status_code, cls)
        except ValueError:
            error_cls = cls
        return error_cls(http_response.info, http_response.content, http_response.request_url, **kwargs)