class NoMockAddress(MockException):
    """The requested URL was not mocked"""

    def __init__(self, request):
        self.request = request

    def __str__(self):
        return 'No mock address: %s %s' % (self.request.method, self.request.url)