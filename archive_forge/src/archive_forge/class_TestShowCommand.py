from cliff import app as application
from cliff import command
from cliff import commandmanager
from cliff import hooks
from cliff import lister
from cliff import show
from cliff.tests import base
from stevedore import extension
from unittest import mock
class TestShowCommand(show.ShowOne):
    """Description of command.
    """

    def take_action(self, parsed_args):
        return (('Name',), ('value',))