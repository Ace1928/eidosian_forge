import os
import sys
import time
from unittest import skipIf
from twisted.internet import abstract, base, defer, error, interfaces, protocol, reactor
from twisted.internet.defer import Deferred, passthru
from twisted.internet.tcp import Connector
from twisted.python import util
from twisted.trial.unittest import TestCase
import %(reactor)s
from twisted.internet import reactor
class ThreePhaseEventTests(TestCase):
    """
    Tests for the private implementation helpers for system event triggers.
    """

    def setUp(self):
        """
        Create a trigger, an argument, and an event to be used by tests.
        """
        self.trigger = lambda x: None
        self.arg = object()
        self.event = base._ThreePhaseEvent()

    def test_addInvalidPhase(self):
        """
        L{_ThreePhaseEvent.addTrigger} should raise L{KeyError} when called
        with an invalid phase.
        """
        self.assertRaises(KeyError, self.event.addTrigger, 'xxx', self.trigger, self.arg)

    def test_addBeforeTrigger(self):
        """
        L{_ThreePhaseEvent.addTrigger} should accept C{'before'} as a phase, a
        callable, and some arguments and add the callable with the arguments to
        the before list.
        """
        self.event.addTrigger('before', self.trigger, self.arg)
        self.assertEqual(self.event.before, [(self.trigger, (self.arg,), {})])

    def test_addDuringTrigger(self):
        """
        L{_ThreePhaseEvent.addTrigger} should accept C{'during'} as a phase, a
        callable, and some arguments and add the callable with the arguments to
        the during list.
        """
        self.event.addTrigger('during', self.trigger, self.arg)
        self.assertEqual(self.event.during, [(self.trigger, (self.arg,), {})])

    def test_addAfterTrigger(self):
        """
        L{_ThreePhaseEvent.addTrigger} should accept C{'after'} as a phase, a
        callable, and some arguments and add the callable with the arguments to
        the after list.
        """
        self.event.addTrigger('after', self.trigger, self.arg)
        self.assertEqual(self.event.after, [(self.trigger, (self.arg,), {})])

    def test_removeTrigger(self):
        """
        L{_ThreePhaseEvent.removeTrigger} should accept an opaque object
        previously returned by L{_ThreePhaseEvent.addTrigger} and remove the
        associated trigger.
        """
        handle = self.event.addTrigger('before', self.trigger, self.arg)
        self.event.removeTrigger(handle)
        self.assertEqual(self.event.before, [])

    def test_removeNonexistentTrigger(self):
        """
        L{_ThreePhaseEvent.removeTrigger} should raise L{ValueError} when given
        an object not previously returned by L{_ThreePhaseEvent.addTrigger}.
        """
        self.assertRaises(ValueError, self.event.removeTrigger, object())

    def test_removeRemovedTrigger(self):
        """
        L{_ThreePhaseEvent.removeTrigger} should raise L{ValueError} the second
        time it is called with an object returned by
        L{_ThreePhaseEvent.addTrigger}.
        """
        handle = self.event.addTrigger('before', self.trigger, self.arg)
        self.event.removeTrigger(handle)
        self.assertRaises(ValueError, self.event.removeTrigger, handle)

    def test_removeAlmostValidTrigger(self):
        """
        L{_ThreePhaseEvent.removeTrigger} should raise L{ValueError} if it is
        given a trigger handle which resembles a valid trigger handle aside
        from its phase being incorrect.
        """
        self.assertRaises(KeyError, self.event.removeTrigger, ('xxx', self.trigger, (self.arg,), {}))

    def test_fireEvent(self):
        """
        L{_ThreePhaseEvent.fireEvent} should call I{before}, I{during}, and
        I{after} phase triggers in that order.
        """
        events = []
        self.event.addTrigger('after', events.append, ('first', 'after'))
        self.event.addTrigger('during', events.append, ('first', 'during'))
        self.event.addTrigger('before', events.append, ('first', 'before'))
        self.event.addTrigger('before', events.append, ('second', 'before'))
        self.event.addTrigger('during', events.append, ('second', 'during'))
        self.event.addTrigger('after', events.append, ('second', 'after'))
        self.assertEqual(events, [])
        self.event.fireEvent()
        self.assertEqual(events, [('first', 'before'), ('second', 'before'), ('first', 'during'), ('second', 'during'), ('first', 'after'), ('second', 'after')])

    def test_asynchronousBefore(self):
        """
        L{_ThreePhaseEvent.fireEvent} should wait for any L{Deferred} returned
        by a I{before} phase trigger before proceeding to I{during} events.
        """
        events = []
        beforeResult = Deferred()
        self.event.addTrigger('before', lambda: beforeResult)
        self.event.addTrigger('during', events.append, 'during')
        self.event.addTrigger('after', events.append, 'after')
        self.assertEqual(events, [])
        self.event.fireEvent()
        self.assertEqual(events, [])
        beforeResult.callback(None)
        self.assertEqual(events, ['during', 'after'])

    def test_beforeTriggerException(self):
        """
        If a before-phase trigger raises a synchronous exception, it should be
        logged and the remaining triggers should be run.
        """
        events = []

        class DummyException(Exception):
            pass

        def raisingTrigger():
            raise DummyException()
        self.event.addTrigger('before', raisingTrigger)
        self.event.addTrigger('before', events.append, 'before')
        self.event.addTrigger('during', events.append, 'during')
        self.event.fireEvent()
        self.assertEqual(events, ['before', 'during'])
        errors = self.flushLoggedErrors(DummyException)
        self.assertEqual(len(errors), 1)

    def test_duringTriggerException(self):
        """
        If a during-phase trigger raises a synchronous exception, it should be
        logged and the remaining triggers should be run.
        """
        events = []

        class DummyException(Exception):
            pass

        def raisingTrigger():
            raise DummyException()
        self.event.addTrigger('during', raisingTrigger)
        self.event.addTrigger('during', events.append, 'during')
        self.event.addTrigger('after', events.append, 'after')
        self.event.fireEvent()
        self.assertEqual(events, ['during', 'after'])
        errors = self.flushLoggedErrors(DummyException)
        self.assertEqual(len(errors), 1)

    def test_synchronousRemoveAlreadyExecutedBefore(self):
        """
        If a before-phase trigger tries to remove another before-phase trigger
        which has already run, a warning should be emitted.
        """
        events = []

        def removeTrigger():
            self.event.removeTrigger(beforeHandle)
        beforeHandle = self.event.addTrigger('before', events.append, ('first', 'before'))
        self.event.addTrigger('before', removeTrigger)
        self.event.addTrigger('before', events.append, ('second', 'before'))
        self.assertWarns(DeprecationWarning, 'Removing already-fired system event triggers will raise an exception in a future version of Twisted.', __file__, self.event.fireEvent)
        self.assertEqual(events, [('first', 'before'), ('second', 'before')])

    def test_synchronousRemovePendingBefore(self):
        """
        If a before-phase trigger removes another before-phase trigger which
        has not yet run, the removed trigger should not be run.
        """
        events = []
        self.event.addTrigger('before', lambda: self.event.removeTrigger(beforeHandle))
        beforeHandle = self.event.addTrigger('before', events.append, ('first', 'before'))
        self.event.addTrigger('before', events.append, ('second', 'before'))
        self.event.fireEvent()
        self.assertEqual(events, [('second', 'before')])

    def test_synchronousBeforeRemovesDuring(self):
        """
        If a before-phase trigger removes a during-phase trigger, the
        during-phase trigger should not be run.
        """
        events = []
        self.event.addTrigger('before', lambda: self.event.removeTrigger(duringHandle))
        duringHandle = self.event.addTrigger('during', events.append, 'during')
        self.event.addTrigger('after', events.append, 'after')
        self.event.fireEvent()
        self.assertEqual(events, ['after'])

    def test_asynchronousBeforeRemovesDuring(self):
        """
        If a before-phase trigger returns a L{Deferred} and later removes a
        during-phase trigger before the L{Deferred} fires, the during-phase
        trigger should not be run.
        """
        events = []
        beforeResult = Deferred()
        self.event.addTrigger('before', lambda: beforeResult)
        duringHandle = self.event.addTrigger('during', events.append, 'during')
        self.event.addTrigger('after', events.append, 'after')
        self.event.fireEvent()
        self.event.removeTrigger(duringHandle)
        beforeResult.callback(None)
        self.assertEqual(events, ['after'])

    def test_synchronousBeforeRemovesConspicuouslySimilarDuring(self):
        """
        If a before-phase trigger removes a during-phase trigger which is
        identical to an already-executed before-phase trigger aside from their
        phases, no warning should be emitted and the during-phase trigger
        should not be run.
        """
        events = []

        def trigger():
            events.append('trigger')
        self.event.addTrigger('before', trigger)
        self.event.addTrigger('before', lambda: self.event.removeTrigger(duringTrigger))
        duringTrigger = self.event.addTrigger('during', trigger)
        self.event.fireEvent()
        self.assertEqual(events, ['trigger'])

    def test_synchronousRemovePendingDuring(self):
        """
        If a during-phase trigger removes another during-phase trigger which
        has not yet run, the removed trigger should not be run.
        """
        events = []
        self.event.addTrigger('during', lambda: self.event.removeTrigger(duringHandle))
        duringHandle = self.event.addTrigger('during', events.append, ('first', 'during'))
        self.event.addTrigger('during', events.append, ('second', 'during'))
        self.event.fireEvent()
        self.assertEqual(events, [('second', 'during')])

    def test_triggersRunOnce(self):
        """
        A trigger should only be called on the first call to
        L{_ThreePhaseEvent.fireEvent}.
        """
        events = []
        self.event.addTrigger('before', events.append, 'before')
        self.event.addTrigger('during', events.append, 'during')
        self.event.addTrigger('after', events.append, 'after')
        self.event.fireEvent()
        self.event.fireEvent()
        self.assertEqual(events, ['before', 'during', 'after'])

    def test_finishedBeforeTriggersCleared(self):
        """
        The temporary list L{_ThreePhaseEvent.finishedBefore} should be emptied
        and the state reset to C{'BASE'} before the first during-phase trigger
        executes.
        """
        events = []

        def duringTrigger():
            events.append('during')
            self.assertEqual(self.event.finishedBefore, [])
            self.assertEqual(self.event.state, 'BASE')
        self.event.addTrigger('before', events.append, 'before')
        self.event.addTrigger('during', duringTrigger)
        self.event.fireEvent()
        self.assertEqual(events, ['before', 'during'])