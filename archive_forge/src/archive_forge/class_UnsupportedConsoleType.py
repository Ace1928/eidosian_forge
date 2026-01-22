class UnsupportedConsoleType(Exception):
    """Indicates that the user is trying to use an unsupported
    console type when retrieving console urls of servers.
    """

    def __init__(self, console_type):
        self.message = 'Unsupported console_type "%s"' % console_type