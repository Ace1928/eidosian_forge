import fixtures
class HttpCheckFixture(fixtures.Fixture):
    """Helps short circuit the external http call"""

    def __init__(self, return_value=True):
        """Initialize the fixture.

        :param return_value: True implies the policy check passed and False
               implies that the policy check failed
        :type return_value: boolean
        """
        super(HttpCheckFixture, self).__init__()
        self.return_value = return_value

    def setUp(self):
        super(HttpCheckFixture, self).setUp()

        def mocked_call(target, cred, enforcer, rule):
            return self.return_value
        self.useFixture(fixtures.MonkeyPatch('oslo_policy._external.HttpCheck.__call__', mocked_call))