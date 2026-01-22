class WebSocketBadStatusException(WebSocketException):
    """
    WebSocketBadStatusException will be raised when we get bad handshake status code.
    """

    def __init__(self, message, status_code, status_message=None, resp_headers=None):
        msg = message % (status_code, status_message)
        super(WebSocketBadStatusException, self).__init__(msg)
        self.status_code = status_code
        self.resp_headers = resp_headers