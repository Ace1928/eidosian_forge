from prov import Error
class DoNotExist(Error):
    """Exception for the case a serializer is not available."""
    pass