from twisted.cred.error import UnauthorizedLogin
class ConchError(Exception):

    def __init__(self, value, data=None):
        Exception.__init__(self, value, data)
        self.value = value
        self.data = data