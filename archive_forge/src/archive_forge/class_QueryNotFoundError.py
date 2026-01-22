class QueryNotFoundError(Error):
    """DEPRECATED: Raised by Iterator methods when the Iterator is invalid. This
  should not happen during normal usage; it protects against malicious users
  and system errors.
  """