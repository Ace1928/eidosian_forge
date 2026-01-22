class UnusedAnnotation(Message):
    """
    Indicates that a variable has been explicitly annotated to but not actually
    used.
    """
    message = 'local variable %r is annotated but never used'

    def __init__(self, filename, loc, names):
        Message.__init__(self, filename, loc)
        self.message_args = (names,)