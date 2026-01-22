class StringDotFormatInvalidFormat(Message):
    message = "'...'.format(...) has invalid format string: %s"

    def __init__(self, filename, loc, error):
        Message.__init__(self, filename, loc)
        self.message_args = (error,)