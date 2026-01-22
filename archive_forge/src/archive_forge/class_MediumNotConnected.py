class MediumNotConnected(InternalBzrError):
    _fmt = "The medium '%(medium)s' is not connected."

    def __init__(self, medium):
        self.medium = medium