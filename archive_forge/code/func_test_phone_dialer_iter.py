import collections
import functools
import random
from automaton import exceptions as excp
from automaton import machines
from automaton import runners
from testtools import testcase
def test_phone_dialer_iter(self):
    dialer, number_calling = self._make_phone_dialer()
    self.assertEqual(0, len(number_calling))
    r = runners.HierarchicalRunner(dialer)
    transitions = list(r.run_iter('dial'))
    self.assertEqual(('talk', 'hangup'), transitions[-1])
    self.assertEqual(len(number_calling), sum((1 if new_state == 'accumulate' else 0 for old_state, new_state in transitions)))
    self.assertEqual(10, len(number_calling))