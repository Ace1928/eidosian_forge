import collections
import functools
import random
from automaton import exceptions as excp
from automaton import machines
from automaton import runners
from testtools import testcase
def test_phone_call_iter(self):
    handler = self._make_phone_call()
    r = runners.HierarchicalRunner(handler)
    transitions = list(r.run_iter('call'))
    self.assertEqual(('talk', 'hangup'), transitions[-1])
    self.assertEqual(('begin', 'phone'), transitions[0])
    talk_talk = 0
    for transition in transitions:
        if transition == ('talk', 'talk'):
            talk_talk += 1
    self.assertGreater(talk_talk, 0)