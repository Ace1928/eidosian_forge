class MalformedError(DefaultCredentialsError, ValueError):
    """An exception for malformed data."""