class WritingNotComplete(InternalBzrError):
    _fmt = "The MediumRequest '%(request)s' has not has finish_writing called upon it - until the write phase is complete no data may be read."

    def __init__(self, request):
        self.request = request