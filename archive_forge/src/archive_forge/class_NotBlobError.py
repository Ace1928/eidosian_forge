import binascii
class NotBlobError(WrongObjectException):
    """Indicates that the sha requested does not point to a blob."""
    type_name = 'blob'