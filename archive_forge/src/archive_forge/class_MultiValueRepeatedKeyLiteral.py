class MultiValueRepeatedKeyLiteral(Message):
    message = 'dictionary key %r repeated with different values'

    def __init__(self, filename, loc, key):
        Message.__init__(self, filename, loc)
        self.message_args = (key,)