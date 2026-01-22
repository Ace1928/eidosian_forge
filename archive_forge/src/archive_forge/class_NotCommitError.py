import binascii
class NotCommitError(WrongObjectException):
    """Indicates that the sha requested does not point to a commit."""
    type_name = 'commit'