import socket
class PartialAuthentication(AuthenticationException):
    """
    An internal exception thrown in the case of partial authentication.
    """
    allowed_types = []

    def __init__(self, types):
        AuthenticationException.__init__(self, types)
        self.allowed_types = types

    def __str__(self):
        return 'Partial authentication; allowed types: {!r}'.format(self.allowed_types)