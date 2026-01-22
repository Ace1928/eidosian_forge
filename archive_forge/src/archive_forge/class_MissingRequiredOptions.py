from keystoneauth1.exceptions import base
class MissingRequiredOptions(OptionError):
    """One or more required options were not provided.

    :param list(keystoneauth1.loading.Opt) options: Missing options.

    .. py:attribute:: options

        List of the missing options.
    """

    def __init__(self, options):
        self.options = options
        names = ', '.join((o.dest for o in options))
        m = 'Auth plugin requires parameters which were not given: %s'
        super(MissingRequiredOptions, self).__init__(m % names)