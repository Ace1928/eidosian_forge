import h2.errors
class InvalidSettingsValueError(ProtocolError, ValueError):
    """
    An attempt was made to set an invalid Settings value.

    .. versionadded:: 2.0.0
    """

    def __init__(self, msg, error_code):
        super(InvalidSettingsValueError, self).__init__(msg)
        self.error_code = error_code