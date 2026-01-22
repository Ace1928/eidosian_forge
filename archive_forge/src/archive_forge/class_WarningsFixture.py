import fixtures
import hashlib
import uuid
import warnings
from oslotest import base as test_base
from oslo_context import context
from oslo_context import fixture
class WarningsFixture(fixtures.Fixture):

    def __init__(self, action='always', category=DeprecationWarning):
        super(WarningsFixture, self).__init__()
        self.action = action
        self.category = category

    def setUp(self):
        super(WarningsFixture, self).setUp()
        self._w = warnings.catch_warnings(record=True)
        self.log = self._w.__enter__()
        self.addCleanup(self._w.__exit__)
        warnings.simplefilter(self.action, self.category)

    def __len__(self):
        return len(self.log)

    def __getitem__(self, item):
        return self.log[item]