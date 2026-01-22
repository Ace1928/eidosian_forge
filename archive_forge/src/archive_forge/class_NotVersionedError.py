class NotVersionedError(BzrError):
    """Used when a path is expected to be versioned, but it is not."""
    _fmt = '%(context_info)s%(path)s is not versioned.'

    def __init__(self, path, context_info=None):
        """Construct a new NotVersionedError.

        :param path: This is the path which is not versioned,
            which should be in a user friendly form.
        :param context_info: If given, this is information about the context,
            which could explain why this is expected to be versioned.
        """
        BzrError.__init__(self)
        self.path = path
        if context_info is None:
            self.context_info = ''
        else:
            self.context_info = context_info + '. '