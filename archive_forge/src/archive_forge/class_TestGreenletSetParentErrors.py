from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gc
import sys
import time
import threading
from abc import ABCMeta, abstractmethod
import greenlet
from greenlet import greenlet as RawGreenlet
from . import TestCase
from .leakcheck import fails_leakcheck
class TestGreenletSetParentErrors(TestCase):

    def test_threaded_reparent(self):
        data = {}
        created_event = threading.Event()
        done_event = threading.Event()

        def run():
            data['g'] = RawGreenlet(lambda: None)
            created_event.set()
            done_event.wait(10)

        def blank():
            greenlet.getcurrent().parent.switch()
        thread = threading.Thread(target=run)
        thread.start()
        created_event.wait(10)
        g = RawGreenlet(blank)
        g.switch()
        with self.assertRaises(ValueError) as exc:
            g.parent = data['g']
        done_event.set()
        thread.join(10)
        self.assertEqual(str(exc.exception), 'parent cannot be on a different thread')

    def test_unexpected_reparenting(self):
        another = []

        def worker():
            g = RawGreenlet(lambda: None)
            another.append(g)
            g.switch()
        t = threading.Thread(target=worker)
        t.start()
        t.join(10)
        self.wait_for_pending_cleanups(initial_main_greenlets=self.main_greenlets_before_test + 1)

        class convoluted(RawGreenlet):

            def __getattribute__(self, name):
                if name == 'run':
                    self.parent = another[0]
                return RawGreenlet.__getattribute__(self, name)
        g = convoluted(lambda: None)
        with self.assertRaises(greenlet.error) as exc:
            g.switch()
        self.assertEqual(str(exc.exception), 'cannot switch to a different thread (which happens to have exited)')
        del another[:]

    def test_unexpected_reparenting_thread_running(self):
        another = []
        switched_to_greenlet = threading.Event()
        keep_main_alive = threading.Event()

        def worker():
            g = RawGreenlet(lambda: None)
            another.append(g)
            g.switch()
            switched_to_greenlet.set()
            keep_main_alive.wait(10)

        class convoluted(RawGreenlet):

            def __getattribute__(self, name):
                if name == 'run':
                    self.parent = another[0]
                return RawGreenlet.__getattribute__(self, name)
        t = threading.Thread(target=worker)
        t.start()
        switched_to_greenlet.wait(10)
        try:
            g = convoluted(lambda: None)
            with self.assertRaises(greenlet.error) as exc:
                g.switch()
            self.assertEqual(str(exc.exception), 'cannot switch to a different thread')
        finally:
            keep_main_alive.set()
            t.join(10)
            del another[:]

    def test_cannot_delete_parent(self):
        worker = RawGreenlet(lambda: None)
        self.assertIs(worker.parent, greenlet.getcurrent())
        with self.assertRaises(AttributeError) as exc:
            del worker.parent
        self.assertEqual(str(exc.exception), "can't delete attribute")

    def test_cannot_delete_parent_of_main(self):
        with self.assertRaises(AttributeError) as exc:
            del greenlet.getcurrent().parent
        self.assertEqual(str(exc.exception), "can't delete attribute")

    def test_main_greenlet_parent_is_none(self):
        self.assertIsNone(greenlet.getcurrent().parent)

    def test_set_parent_wrong_types(self):

        def bg():
            greenlet.getcurrent().parent.switch()

        def check(glet):
            for p in (None, 1, self, '42'):
                with self.assertRaises(TypeError) as exc:
                    glet.parent = p
                self.assertEqual(str(exc.exception), 'GreenletChecker: Expected any type of greenlet, not ' + type(p).__name__)
        g = RawGreenlet(bg)
        self.assertFalse(g)
        check(g)
        g.switch()
        self.assertTrue(g)
        check(g)
        g.switch()

    def test_trivial_cycle(self):
        glet = RawGreenlet(lambda: None)
        with self.assertRaises(ValueError) as exc:
            glet.parent = glet
        self.assertEqual(str(exc.exception), 'cyclic parent chain')

    def test_trivial_cycle_main(self):
        with self.assertRaises(AttributeError) as exc:
            greenlet.getcurrent().parent = greenlet.getcurrent()
        self.assertEqual(str(exc.exception), 'cannot set the parent of a main greenlet')

    def test_deeper_cycle(self):
        g1 = RawGreenlet(lambda: None)
        g2 = RawGreenlet(lambda: None)
        g3 = RawGreenlet(lambda: None)
        g1.parent = g2
        g2.parent = g3
        with self.assertRaises(ValueError) as exc:
            g3.parent = g1
        self.assertEqual(str(exc.exception), 'cyclic parent chain')