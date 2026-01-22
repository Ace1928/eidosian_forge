class UnknownFeature(ImportError):
    """Raised when an unknown feature is given in the input stream."""
    _fmt = "Unknown feature '%(feature)s' - try a later importer or an earlier data format"

    def __init__(self, feature):
        self.feature = feature
        ImportError.__init__(self)