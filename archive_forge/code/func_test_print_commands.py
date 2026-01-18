import collections
import json
import optparse
import sys
from unittest import mock
import testtools
from troveclient.compat import common
def test_print_commands(self):
    commands = {'cmd-1': 'cmd 1', 'cmd-2': 'cmd 2'}
    common.print_commands(commands)
    pass