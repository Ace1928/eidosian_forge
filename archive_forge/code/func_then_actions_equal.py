from unittest import mock
from cliff import app as application
from cliff import commandmanager
from cliff import complete
from cliff.tests import base
def then_actions_equal(self, actions):
    optstr = ' '.join((opt for action in actions for opt in action.option_strings))
    self.assertEqual('-h --help --name --shell', optstr)