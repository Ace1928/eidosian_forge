import collections
import functools
import random
from automaton import exceptions as excp
from automaton import machines
from automaton import runners
from testtools import testcase
def test_not_initialized(self):
    self.assertRaises(excp.NotInitialized, self.jumper.process_event, 'jump')