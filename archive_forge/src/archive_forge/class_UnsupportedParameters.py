from keystoneauth1.exceptions import base
class UnsupportedParameters(AuthPluginException):
    """A parameter that was provided or returned is not supported.

    :param list(str) names: Names of the unsupported parameters.

    .. py:attribute:: names

        Names of the unsupported parameters.
    """

    def __init__(self, names):
        self.names = names
        m = 'The following parameters were given that are unsupported: %s'
        super(UnsupportedParameters, self).__init__(m % ', '.join(self.names))