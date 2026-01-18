import collections
import json
import optparse
import sys
from unittest import mock
import testtools
from troveclient.compat import common
def test_print_actions(self):
    cmd = 'test-cmd'
    actions = {'test': 'test action', 'help': 'help action'}
    common.print_actions(cmd, actions)
    pass