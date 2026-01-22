class PlotlyDictValueError(PlotlyGraphObjectError):

    def __init__(self, obj, path, notes=()):
        """See PlotlyGraphObjectError.__init__ for param docs."""
        format_dict = {'attribute': path[-1], 'object_name': obj._name}
        message = "'{attribute}' has invalid value inside '{object_name}'".format(**format_dict)
        notes = [obj.help(path[-1], return_help=True)] + list(notes)
        super(PlotlyDictValueError, self).__init__(message=message, notes=notes, path=path)