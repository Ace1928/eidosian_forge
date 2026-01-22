class BadFilterError(Error):
    """Raised by Query.__setitem__() and Query.Run() when a filter string is
  invalid.
  """

    def __init__(self, filter):
        self.filter = filter
        message = (u'invalid filter: %s.' % self.filter).encode('utf-8')
        super(BadFilterError, self).__init__(message)