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
class TestGreenlet(TestCase):

    def _do_simple_test(self):
        lst = []

        def f():
            lst.append(1)
            greenlet.getcurrent().parent.switch()
            lst.append(3)
        g = RawGreenlet(f)
        lst.append(0)
        g.switch()
        lst.append(2)
        g.switch()
        lst.append(4)
        self.assertEqual(lst, list(range(5)))

    def test_simple(self):
        self._do_simple_test()

    def test_switch_no_run_raises_AttributeError(self):
        g = RawGreenlet()
        with self.assertRaises(AttributeError) as exc:
            g.switch()
        self.assertIn('run', str(exc.exception))

    def test_throw_no_run_raises_AttributeError(self):
        g = RawGreenlet()
        with self.assertRaises(AttributeError) as exc:
            g.throw(SomeError)
        self.assertIn('run', str(exc.exception))

    def test_parent_equals_None(self):
        g = RawGreenlet(parent=None)
        self.assertIsNotNone(g)
        self.assertIs(g.parent, greenlet.getcurrent())

    def test_run_equals_None(self):
        g = RawGreenlet(run=None)
        self.assertIsNotNone(g)
        self.assertIsNone(g.run)

    def test_two_children(self):
        lst = []

        def f():
            lst.append(1)
            greenlet.getcurrent().parent.switch()
            lst.extend([1, 1])
        g = RawGreenlet(f)
        h = RawGreenlet(f)
        g.switch()
        self.assertEqual(len(lst), 1)
        h.switch()
        self.assertEqual(len(lst), 2)
        h.switch()
        self.assertEqual(len(lst), 4)
        self.assertEqual(h.dead, True)
        g.switch()
        self.assertEqual(len(lst), 6)
        self.assertEqual(g.dead, True)

    def test_two_recursive_children(self):
        lst = []

        def f():
            lst.append('b')
            greenlet.getcurrent().parent.switch()

        def g():
            lst.append('a')
            g = RawGreenlet(f)
            g.switch()
            lst.append('c')
        g = RawGreenlet(g)
        self.assertEqual(sys.getrefcount(g), 2)
        g.switch()
        self.assertEqual(lst, ['a', 'b', 'c'])
        self.assertEqual(sys.getrefcount(g), 2)

    def test_threads(self):
        success = []

        def f():
            self._do_simple_test()
            success.append(True)
        ths = [threading.Thread(target=f) for i in range(10)]
        for th in ths:
            th.start()
        for th in ths:
            th.join(10)
        self.assertEqual(len(success), len(ths))

    def test_exception(self):
        seen = []
        g1 = RawGreenlet(fmain)
        g2 = RawGreenlet(fmain)
        g1.switch(seen)
        g2.switch(seen)
        g2.parent = g1
        self.assertEqual(seen, [])
        self.assertRaises(SomeError, g2.switch)
        self.assertEqual(seen, [SomeError])
        value = g2.switch()
        self.assertEqual(value, ())
        self.assertEqual(seen, [SomeError])
        value = g2.switch(25)
        self.assertEqual(value, 25)
        self.assertEqual(seen, [SomeError])

    def test_send_exception(self):
        seen = []
        g1 = RawGreenlet(fmain)
        g1.switch(seen)
        self.assertRaises(KeyError, send_exception, g1, KeyError)
        self.assertEqual(seen, [KeyError])

    def test_dealloc(self):
        seen = []
        g1 = RawGreenlet(fmain)
        g2 = RawGreenlet(fmain)
        g1.switch(seen)
        g2.switch(seen)
        self.assertEqual(seen, [])
        del g1
        gc.collect()
        self.assertEqual(seen, [greenlet.GreenletExit])
        del g2
        gc.collect()
        self.assertEqual(seen, [greenlet.GreenletExit, greenlet.GreenletExit])

    def test_dealloc_catches_GreenletExit_throws_other(self):

        def run():
            try:
                greenlet.getcurrent().parent.switch()
            except greenlet.GreenletExit:
                raise SomeError from None
        g = RawGreenlet(run)
        g.switch()
        oldstderr = sys.stderr
        try:
            from cStringIO import StringIO
        except ImportError:
            from io import StringIO
        stderr = sys.stderr = StringIO()
        try:
            del g
        finally:
            sys.stderr = oldstderr
        v = stderr.getvalue()
        self.assertIn('Exception', v)
        self.assertIn('ignored', v)
        self.assertIn('SomeError', v)

    def test_dealloc_other_thread(self):
        seen = []
        someref = []
        bg_glet_created_running_and_no_longer_ref_in_bg = threading.Event()
        fg_ref_released = threading.Event()
        bg_should_be_clear = threading.Event()
        ok_to_exit_bg_thread = threading.Event()

        def f():
            g1 = RawGreenlet(fmain)
            g1.switch(seen)
            someref.append(g1)
            del g1
            gc.collect()
            bg_glet_created_running_and_no_longer_ref_in_bg.set()
            fg_ref_released.wait(3)
            RawGreenlet()
            bg_should_be_clear.set()
            ok_to_exit_bg_thread.wait(3)
            RawGreenlet()
        t = threading.Thread(target=f)
        t.start()
        bg_glet_created_running_and_no_longer_ref_in_bg.wait(10)
        self.assertEqual(seen, [])
        self.assertEqual(len(someref), 1)
        del someref[:]
        gc.collect()
        self.assertEqual(seen, [])
        fg_ref_released.set()
        bg_should_be_clear.wait(3)
        try:
            self.assertEqual(seen, [greenlet.GreenletExit])
        finally:
            ok_to_exit_bg_thread.set()
            t.join(10)
            del seen[:]
            del someref[:]

    def test_frame(self):

        def f1():
            f = sys._getframe(0)
            self.assertEqual(f.f_back, None)
            greenlet.getcurrent().parent.switch(f)
            return 'meaning of life'
        g = RawGreenlet(f1)
        frame = g.switch()
        self.assertTrue(frame is g.gr_frame)
        self.assertTrue(g)
        from_g = g.switch()
        self.assertFalse(g)
        self.assertEqual(from_g, 'meaning of life')
        self.assertEqual(g.gr_frame, None)

    def test_thread_bug(self):

        def runner(x):
            g = RawGreenlet(lambda: time.sleep(x))
            g.switch()
        t1 = threading.Thread(target=runner, args=(0.2,))
        t2 = threading.Thread(target=runner, args=(0.3,))
        t1.start()
        t2.start()
        t1.join(10)
        t2.join(10)

    def test_switch_kwargs(self):

        def run(a, b):
            self.assertEqual(a, 4)
            self.assertEqual(b, 2)
            return 42
        x = RawGreenlet(run).switch(a=4, b=2)
        self.assertEqual(x, 42)

    def test_switch_kwargs_to_parent(self):

        def run(x):
            greenlet.getcurrent().parent.switch(x=x)
            greenlet.getcurrent().parent.switch(2, x=3)
            return (x, x ** 2)
        g = RawGreenlet(run)
        self.assertEqual({'x': 3}, g.switch(3))
        self.assertEqual(((2,), {'x': 3}), g.switch())
        self.assertEqual((3, 9), g.switch())

    def test_switch_to_another_thread(self):
        data = {}
        created_event = threading.Event()
        done_event = threading.Event()

        def run():
            data['g'] = RawGreenlet(lambda: None)
            created_event.set()
            done_event.wait(10)
        thread = threading.Thread(target=run)
        thread.start()
        created_event.wait(10)
        with self.assertRaises(greenlet.error):
            data['g'].switch()
        done_event.set()
        thread.join(10)
        data.clear()

    def test_exc_state(self):

        def f():
            try:
                raise ValueError('fun')
            except:
                exc_info = sys.exc_info()
                RawGreenlet(h).switch()
                self.assertEqual(exc_info, sys.exc_info())

        def h():
            self.assertEqual(sys.exc_info(), (None, None, None))
        RawGreenlet(f).switch()

    def test_instance_dict(self):

        def f():
            greenlet.getcurrent().test = 42

        def deldict(g):
            del g.__dict__

        def setdict(g, value):
            g.__dict__ = value
        g = RawGreenlet(f)
        self.assertEqual(g.__dict__, {})
        g.switch()
        self.assertEqual(g.test, 42)
        self.assertEqual(g.__dict__, {'test': 42})
        g.__dict__ = g.__dict__
        self.assertEqual(g.__dict__, {'test': 42})
        self.assertRaises(TypeError, deldict, g)
        self.assertRaises(TypeError, setdict, g, 42)

    def test_running_greenlet_has_no_run(self):
        has_run = []

        def func():
            has_run.append(hasattr(greenlet.getcurrent(), 'run'))
        g = RawGreenlet(func)
        g.switch()
        self.assertEqual(has_run, [False])

    def test_deepcopy(self):
        import copy
        self.assertRaises(TypeError, copy.copy, RawGreenlet())
        self.assertRaises(TypeError, copy.deepcopy, RawGreenlet())

    def test_parent_restored_on_kill(self):
        hub = RawGreenlet(lambda: None)
        main = greenlet.getcurrent()
        result = []

        def worker():
            try:
                main.switch()
            except greenlet.GreenletExit:
                result.append(greenlet.getcurrent().parent)
                result.append(greenlet.getcurrent())
                hub.switch()
        g = RawGreenlet(worker, parent=hub)
        g.switch()
        del g
        self.assertTrue(result)
        self.assertIs(result[0], main)
        self.assertIs(result[1].parent, hub)
        del result[:]
        hub = None
        main = None

    def test_parent_return_failure(self):
        g1 = RawGreenlet()
        g2 = RawGreenlet(lambda: None, parent=g1)
        with self.assertRaises(AttributeError):
            g2.switch()

    def test_throw_exception_not_lost(self):

        class mygreenlet(RawGreenlet):

            def __getattribute__(self, name):
                try:
                    raise Exception
                except:
                    pass
                return RawGreenlet.__getattribute__(self, name)
        g = mygreenlet(lambda: None)
        self.assertRaises(SomeError, g.throw, SomeError())

    @fails_leakcheck
    def _do_test_throw_to_dead_thread_doesnt_crash(self, wait_for_cleanup=False):
        result = []

        def worker():
            greenlet.getcurrent().parent.switch()

        def creator():
            g = RawGreenlet(worker)
            g.switch()
            result.append(g)
            if wait_for_cleanup:
                g.switch()
                greenlet.getcurrent()
        t = threading.Thread(target=creator)
        t.start()
        t.join(10)
        del t
        if wait_for_cleanup:
            self.wait_for_pending_cleanups()
        with self.assertRaises(greenlet.error) as exc:
            result[0].throw(SomeError)
        if not wait_for_cleanup:
            self.assertIn(str(exc.exception), ['cannot switch to a different thread (which happens to have exited)', 'cannot switch to a different thread'])
        else:
            self.assertEqual(str(exc.exception), 'cannot switch to a different thread (which happens to have exited)')
        if hasattr(result[0].gr_frame, 'clear'):
            with self.assertRaises(RuntimeError):
                result[0].gr_frame.clear()
        if not wait_for_cleanup:
            result[0].gr_frame.f_locals.clear()
        else:
            self.assertIsNone(result[0].gr_frame)
        del creator
        worker = None
        del result[:]
        self.expect_greenlet_leak = True

    @fails_leakcheck
    def test_throw_to_dead_thread_doesnt_crash(self):
        self._do_test_throw_to_dead_thread_doesnt_crash()

    def test_throw_to_dead_thread_doesnt_crash_wait(self):
        self._do_test_throw_to_dead_thread_doesnt_crash(True)

    @fails_leakcheck
    def test_recursive_startup(self):

        class convoluted(RawGreenlet):

            def __init__(self):
                RawGreenlet.__init__(self)
                self.count = 0

            def __getattribute__(self, name):
                if name == 'run' and self.count == 0:
                    self.count = 1
                    self.switch(43)
                return RawGreenlet.__getattribute__(self, name)

            def run(self, value):
                while True:
                    self.parent.switch(value)
        g = convoluted()
        self.assertEqual(g.switch(42), 43)
        self.expect_greenlet_leak = True

    def test_threaded_updatecurrent(self):
        lock1 = threading.Lock()
        lock1.acquire()
        lock2 = threading.Lock()
        lock2.acquire()

        class finalized(object):

            def __del__(self):
                lock2.release()
                lock1.acquire()

        def deallocator():
            greenlet.getcurrent().parent.switch()

        def fthread():
            lock2.acquire()
            greenlet.getcurrent()
            del g[0]
            lock1.release()
            lock2.acquire()
            greenlet.getcurrent()
            lock1.release()
        main = greenlet.getcurrent()
        g = [RawGreenlet(deallocator)]
        g[0].bomb = finalized()
        g[0].switch()
        t = threading.Thread(target=fthread)
        t.start()
        lock2.release()
        lock1.acquire()
        self.assertEqual(greenlet.getcurrent(), main)
        t.join(10)

    def test_dealloc_switch_args_not_lost(self):
        seen = []

        def worker():
            value = greenlet.getcurrent().parent.switch()
            del worker[0]
            initiator.parent = greenlet.getcurrent().parent
            try:
                greenlet.getcurrent().parent.switch(value)
            finally:
                seen.append(greenlet.getcurrent())

        def initiator():
            return 42
        worker = [RawGreenlet(worker)]
        worker[0].switch()
        initiator = RawGreenlet(initiator, worker[0])
        value = initiator.switch()
        self.assertTrue(seen)
        self.assertEqual(value, 42)

    def test_tuple_subclass(self):

        def _apply(func, a, k):
            func(*a, **k)

        class mytuple(tuple):

            def __len__(self):
                greenlet.getcurrent().switch()
                return tuple.__len__(self)
        args = mytuple()
        kwargs = dict(a=42)

        def switchapply():
            _apply(greenlet.getcurrent().parent.switch, args, kwargs)
        g = RawGreenlet(switchapply)
        self.assertEqual(g.switch(), kwargs)

    def test_abstract_subclasses(self):
        AbstractSubclass = ABCMeta('AbstractSubclass', (RawGreenlet,), {'run': abstractmethod(lambda self: None)})

        class BadSubclass(AbstractSubclass):
            pass

        class GoodSubclass(AbstractSubclass):

            def run(self):
                pass
        GoodSubclass()
        self.assertRaises(TypeError, BadSubclass)

    def test_implicit_parent_with_threads(self):
        if not gc.isenabled():
            return
        N = gc.get_threshold()[0]
        if N < 50:
            return

        def attempt():
            lock1 = threading.Lock()
            lock1.acquire()
            lock2 = threading.Lock()
            lock2.acquire()
            recycled = [False]

            def another_thread():
                lock1.acquire()
                greenlet.getcurrent()
                lock2.release()
            t = threading.Thread(target=another_thread)
            t.start()

            class gc_callback(object):

                def __del__(self):
                    lock1.release()
                    lock2.acquire()
                    recycled[0] = True

            class garbage(object):

                def __init__(self):
                    self.cycle = self
                    self.callback = gc_callback()
            l = []
            x = range(N * 2)
            current = greenlet.getcurrent()
            g = garbage()
            for _ in x:
                g = None
                if recycled[0]:
                    t.join(10)
                    return False
                last = RawGreenlet()
                if recycled[0]:
                    break
                l.append(last)
            else:
                gc.collect()
                if recycled[0]:
                    t.join(10)
                return False
            self.assertEqual(last.parent, current)
            for g in l:
                self.assertEqual(g.parent, current)
            return True
        for _ in range(5):
            if attempt():
                break

    def test_issue_245_reference_counting_subclass_no_threads(self):
        from greenlet import getcurrent
        from greenlet import GreenletExit

        class Greenlet(RawGreenlet):
            pass
        initial_refs = sys.getrefcount(Greenlet)
        self.glets = []

        def greenlet_main():
            try:
                getcurrent().parent.switch()
            except GreenletExit:
                self.glets.append(getcurrent())
        for _ in range(10):
            Greenlet(greenlet_main).switch()
        del self.glets
        self.assertEqual(sys.getrefcount(Greenlet), initial_refs)

    def test_issue_245_reference_counting_subclass_threads(self):
        from threading import Thread
        from threading import Event
        from greenlet import getcurrent

        class MyGreenlet(RawGreenlet):
            pass
        glets = []
        ref_cleared = Event()

        def greenlet_main():
            getcurrent().parent.switch()

        def thread_main(greenlet_running_event):
            mine = MyGreenlet(greenlet_main)
            glets.append(mine)
            mine.switch()
            del mine
            greenlet_running_event.set()
            ref_cleared.wait(10)
            getcurrent()
        initial_refs = sys.getrefcount(MyGreenlet)
        thread_ready_events = []
        for _ in range(initial_refs + 45):
            event = Event()
            thread = Thread(target=thread_main, args=(event,))
            thread_ready_events.append(event)
            thread.start()
        for done_event in thread_ready_events:
            done_event.wait(10)
        del glets[:]
        ref_cleared.set()
        self.wait_for_pending_cleanups()
        self.assertEqual(sys.getrefcount(MyGreenlet), initial_refs)

    def test_falling_off_end_switches_to_unstarted_parent_raises_error(self):

        def no_args():
            return 13
        parent_never_started = RawGreenlet(no_args)

        def leaf():
            return 42
        child = RawGreenlet(leaf, parent_never_started)
        with self.assertRaises(TypeError):
            child.switch()

    def test_falling_off_end_switches_to_unstarted_parent_works(self):

        def one_arg(x):
            return (x, 24)
        parent_never_started = RawGreenlet(one_arg)

        def leaf():
            return 42
        child = RawGreenlet(leaf, parent_never_started)
        result = child.switch()
        self.assertEqual(result, (42, 24))

    def test_switch_to_dead_greenlet_with_unstarted_perverse_parent(self):

        class Parent(RawGreenlet):

            def __getattribute__(self, name):
                if name == 'run':
                    raise SomeError
        parent_never_started = Parent()
        seen = []
        child = RawGreenlet(lambda: seen.append(42), parent_never_started)
        with self.assertRaises(SomeError):
            child.switch()
        self.assertEqual(seen, [42])
        with self.assertRaises(SomeError):
            child.switch()
        self.assertEqual(seen, [42])

    def test_switch_to_dead_greenlet_reparent(self):
        seen = []
        parent_never_started = RawGreenlet(lambda: seen.append(24))
        child = RawGreenlet(lambda: seen.append(42))
        child.switch()
        self.assertEqual(seen, [42])
        child.parent = parent_never_started
        result = child.switch()
        self.assertIsNone(result)
        self.assertEqual(seen, [42, 24])

    def test_can_access_f_back_of_suspended_greenlet(self):
        main = greenlet.getcurrent()

        def outer():
            inner()

        def inner():
            main.switch(sys._getframe(0))
        hub = RawGreenlet(outer)
        hub.switch()
        unrelated = RawGreenlet(lambda: None)
        unrelated.switch()
        self.assertIsNotNone(hub.gr_frame)
        self.assertEqual(hub.gr_frame.f_code.co_name, 'inner')
        self.assertIsNotNone(hub.gr_frame.f_back)
        self.assertEqual(hub.gr_frame.f_back.f_code.co_name, 'outer')
        self.assertIsNone(hub.gr_frame.f_back.f_back)

    def test_get_stack_with_nested_c_calls(self):
        from functools import partial
        from . import _test_extension_cpp

        def recurse(v):
            if v > 0:
                return v * _test_extension_cpp.test_call(partial(recurse, v - 1))
            return greenlet.getcurrent().parent.switch()
        gr = RawGreenlet(recurse)
        gr.switch(5)
        frame = gr.gr_frame
        for i in range(5):
            self.assertEqual(frame.f_locals['v'], i)
            frame = frame.f_back
        self.assertEqual(frame.f_locals['v'], 5)
        self.assertIsNone(frame.f_back)
        self.assertEqual(gr.switch(10), 1200)

    def test_frames_always_exposed(self):
        main = greenlet.getcurrent()

        def outer():
            inner(sys._getframe(0))

        def inner(frame):
            main.switch(frame)
        gr = RawGreenlet(outer)
        frame = gr.switch()
        unrelated = RawGreenlet(lambda: None)
        unrelated.switch()
        self.assertEqual(frame.f_code.co_name, 'outer')
        self.assertIsNone(frame.f_back)