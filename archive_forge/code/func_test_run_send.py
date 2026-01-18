import collections
import functools
import random
from automaton import exceptions as excp
from automaton import machines
from automaton import runners
from testtools import testcase
def test_run_send(self):
    up_downs = []
    runner = runners.FiniteRunner(self.jumper)
    it = runner.run_iter('jump')
    while True:
        up_downs.append(it.send(None))
        if len(up_downs) >= 3:
            it.close()
            break
    self.assertEqual('up', self.jumper.current_state)
    self.assertFalse(self.jumper.terminated)
    self.assertEqual([('down', 'up'), ('up', 'down'), ('down', 'up')], up_downs)
    self.assertRaises(StopIteration, next, it)