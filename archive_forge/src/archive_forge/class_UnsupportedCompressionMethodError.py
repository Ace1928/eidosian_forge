class UnsupportedCompressionMethodError(ArchiveError):
    """Exception raised for unsupported compression parameter given.

    Attributes:
      data -- unknown property data
      message -- explanation of error
    """

    def __init__(self, data, message):
        super().__init__(data, message)
        self.data = data
        self.message = message