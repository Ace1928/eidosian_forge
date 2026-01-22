class ReadingCompleted(InternalBzrError):
    _fmt = "The MediumRequest '%(request)s' has already had finish_reading called upon it - the request has been completed and no more data may be read."

    def __init__(self, request):
        self.request = request