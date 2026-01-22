import logging
from unittest import mock
from osc_lib import logs
from osc_lib.tests import utils
class TestFileFormatter(utils.TestCase):

    def test_nothing(self):
        formatter = logs._FileFormatter()
        self.assertEqual('%(asctime)s.%(msecs)03d %(process)d %(levelname)s %(name)s %(message)s', formatter.fmt)

    def test_options(self):

        class Opts(object):
            cloud = 'cloudy'
            os_project_name = 'projecty'
            username = 'usernamey'
        options = Opts()
        formatter = logs._FileFormatter(options=options)
        self.assertEqual('%(asctime)s.%(msecs)03d %(process)d %(levelname)s %(name)s [cloudy usernamey projecty] %(message)s', formatter.fmt)

    def test_config(self):
        config = mock.Mock()
        config.config = {'cloud': 'cloudy'}
        config.auth = {'project_name': 'projecty', 'username': 'usernamey'}
        formatter = logs._FileFormatter(config=config)
        self.assertEqual('%(asctime)s.%(msecs)03d %(process)d %(levelname)s %(name)s [cloudy usernamey projecty] %(message)s', formatter.fmt)