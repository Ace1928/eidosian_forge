class WritingCompleted(InternalBzrError):
    _fmt = "The MediumRequest '%(request)s' has already had finish_writing called upon it - accept bytes may not be called anymore."

    def __init__(self, request):
        self.request = request