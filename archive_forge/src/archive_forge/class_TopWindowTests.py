from __future__ import annotations
from typing import Callable
from twisted.conch.insults.window import ScrolledArea, TextOutput, TopWindow
from twisted.trial.unittest import TestCase
class TopWindowTests(TestCase):
    """
    Tests for L{TopWindow}, the root window container class.
    """

    def test_paintScheduling(self) -> None:
        """
        Verify that L{TopWindow.repaint} schedules an actual paint to occur
        using the scheduling object passed to its initializer.
        """
        paints: list[None] = []
        scheduled: list[Callable[[], object]] = []
        root = TopWindow(lambda: paints.append(None), scheduled.append)
        self.assertEqual(paints, [])
        self.assertEqual(scheduled, [])
        root.repaint()
        self.assertEqual(paints, [])
        self.assertEqual(len(scheduled), 1)
        root.repaint()
        self.assertEqual(paints, [])
        self.assertEqual(len(scheduled), 1)
        scheduled.pop()()
        self.assertEqual(len(paints), 1)
        self.assertEqual(scheduled, [])
        root.repaint()
        self.assertEqual(len(paints), 1)
        self.assertEqual(len(scheduled), 1)