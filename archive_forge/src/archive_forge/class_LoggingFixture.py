import fixtures
class LoggingFixture(fixtures.Fixture):

    def __init__(self, suffix='', calls=None):
        super(LoggingFixture, self).__init__()
        if calls is None:
            calls = []
        self.calls = calls
        self.suffix = suffix

    def setUp(self):
        super(LoggingFixture, self).setUp()
        self.calls.append('setUp' + self.suffix)
        self.addCleanup(self.calls.append, 'cleanUp' + self.suffix)

    def reset(self):
        self.calls.append('reset' + self.suffix)