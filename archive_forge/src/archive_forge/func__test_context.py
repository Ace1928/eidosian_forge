from __future__ import print_function
import gc
import sys
import unittest
from functools import partial
from unittest import skipUnless
from unittest import skipIf
from greenlet import greenlet
from greenlet import getcurrent
from . import TestCase
def _test_context(self, propagate_by):
    ID_VAR.set(0)
    callback = getcurrent().switch
    counts = dict(((i, 0) for i in range(5)))
    lets = [greenlet(partial(partial(copy_context().run, self._increment) if propagate_by == 'run' else self._increment, greenlet_id=i, callback=callback, counts=counts, expect=i - 1 if propagate_by == 'share' else 0 if propagate_by in ('set', 'run') else None)) for i in range(1, 5)]
    for let in lets:
        if propagate_by == 'set':
            let.gr_context = copy_context()
        elif propagate_by == 'share':
            let.gr_context = getcurrent().gr_context
    for i in range(2):
        counts[ID_VAR.get()] += 1
        for let in lets:
            let.switch()
    if propagate_by == 'run':
        for let in reversed(lets):
            let.switch()
    else:
        for let in lets:
            let.switch()
    for let in lets:
        self.assertTrue(let.dead)
        if propagate_by == 'run':
            self.assertIsNone(let.gr_context)
        else:
            self.assertIsNotNone(let.gr_context)
    if propagate_by == 'share':
        self.assertEqual(counts, {0: 1, 1: 1, 2: 1, 3: 1, 4: 6})
    else:
        self.assertEqual(set(counts.values()), set([2]))