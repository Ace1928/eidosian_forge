class PercentFormatPositionalCountMismatch(Message):
    message = "'...' %% ... has %d placeholder(s) but %d substitution(s)"

    def __init__(self, filename, loc, n_placeholders, n_substitutions):
        Message.__init__(self, filename, loc)
        self.message_args = (n_placeholders, n_substitutions)