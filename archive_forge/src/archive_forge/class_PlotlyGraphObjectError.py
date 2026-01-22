class PlotlyGraphObjectError(PlotlyError):

    def __init__(self, message='', path=(), notes=()):
        """
        General graph object error for validation failures.

        :param (str|unicode) message: The error message.
        :param (iterable) path: A path pointing to the error.
        :param notes: Add additional notes, but keep default exception message.

        """
        self.message = message
        self.plain_message = message
        self.path = list(path)
        self.notes = notes
        super(PlotlyGraphObjectError, self).__init__(message)

    def __str__(self):
        """This is called by Python to present the error message."""
        format_dict = {'message': self.message, 'path': '[' + ']['.join((repr(k) for k in self.path)) + ']', 'notes': '\n'.join(self.notes)}
        return '{message}\n\nPath To Error: {path}\n\n{notes}'.format(**format_dict)