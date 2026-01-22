class TooManyHttpHeaders(Error):
    """Raised when a handler specified too many HTTP headers.

  The message should indicate the maximum number of headers allowed.
  """