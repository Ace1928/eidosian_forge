class EmbeddedModuleFailure(Exception):

    def __init__(self, msg, **kwargs):
        self._message = msg
        self._kwargs = kwargs

    def get_message(self):
        return self._message

    @property
    def kwargs(self):
        return self._kwargs

    def __repr__(self):
        return repr(self.get_message())

    def __str__(self):
        return str(self.get_message())