class UnexpectedResponseError(ResponseError):
    """An unexpected response was received."""

    def __str__(self):
        return '%s: %s' % (self.response.status, self.response.reason)