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
class SystemEventTests(TestCase):
    """
    Tests for the reactor's implementation of the C{fireSystemEvent},
    C{addSystemEventTrigger}, and C{removeSystemEventTrigger} methods of the
    L{IReactorCore} interface.

    @ivar triggers: A list of the handles to triggers which have been added to
        the reactor.
    """

    def setUp(self):
        """
        Create an empty list in which to store trigger handles.
        """
        self.triggers = []

    def tearDown(self):
        """
        Remove all remaining triggers from the reactor.
        """
        while self.triggers:
            trigger = self.triggers.pop()
            try:
                reactor.removeSystemEventTrigger(trigger)
            except (ValueError, KeyError):
                pass

    def addTrigger(self, event, phase, func):
        """
        Add a trigger to the reactor and remember it in C{self.triggers}.
        """
        t = reactor.addSystemEventTrigger(event, phase, func)
        self.triggers.append(t)
        return t

    def removeTrigger(self, trigger):
        """
        Remove a trigger by its handle from the reactor and from
        C{self.triggers}.
        """
        reactor.removeSystemEventTrigger(trigger)
        self.triggers.remove(trigger)

    def _addSystemEventTriggerTest(self, phase):
        eventType = 'test'
        events = []

        def trigger():
            events.append(None)
        self.addTrigger(phase, eventType, trigger)
        self.assertEqual(events, [])
        reactor.fireSystemEvent(eventType)
        self.assertEqual(events, [None])

    def test_beforePhase(self):
        """
        L{IReactorCore.addSystemEventTrigger} should accept the C{'before'}
        phase and not call the given object until the right event is fired.
        """
        self._addSystemEventTriggerTest('before')

    def test_duringPhase(self):
        """
        L{IReactorCore.addSystemEventTrigger} should accept the C{'during'}
        phase and not call the given object until the right event is fired.
        """
        self._addSystemEventTriggerTest('during')

    def test_afterPhase(self):
        """
        L{IReactorCore.addSystemEventTrigger} should accept the C{'after'}
        phase and not call the given object until the right event is fired.
        """
        self._addSystemEventTriggerTest('after')

    def test_unknownPhase(self):
        """
        L{IReactorCore.addSystemEventTrigger} should reject phases other than
        C{'before'}, C{'during'}, or C{'after'}.
        """
        eventType = 'test'
        self.assertRaises(KeyError, self.addTrigger, 'xxx', eventType, lambda: None)

    def test_beforePreceedsDuring(self):
        """
        L{IReactorCore.addSystemEventTrigger} should call triggers added to the
        C{'before'} phase before it calls triggers added to the C{'during'}
        phase.
        """
        eventType = 'test'
        events = []

        def beforeTrigger():
            events.append('before')

        def duringTrigger():
            events.append('during')
        self.addTrigger('before', eventType, beforeTrigger)
        self.addTrigger('during', eventType, duringTrigger)
        self.assertEqual(events, [])
        reactor.fireSystemEvent(eventType)
        self.assertEqual(events, ['before', 'during'])

    def test_duringPreceedsAfter(self):
        """
        L{IReactorCore.addSystemEventTrigger} should call triggers added to the
        C{'during'} phase before it calls triggers added to the C{'after'}
        phase.
        """
        eventType = 'test'
        events = []

        def duringTrigger():
            events.append('during')

        def afterTrigger():
            events.append('after')
        self.addTrigger('during', eventType, duringTrigger)
        self.addTrigger('after', eventType, afterTrigger)
        self.assertEqual(events, [])
        reactor.fireSystemEvent(eventType)
        self.assertEqual(events, ['during', 'after'])

    def test_beforeReturnsDeferred(self):
        """
        If a trigger added to the C{'before'} phase of an event returns a
        L{Deferred}, the C{'during'} phase should be delayed until it is called
        back.
        """
        triggerDeferred = Deferred()
        eventType = 'test'
        events = []

        def beforeTrigger():
            return triggerDeferred

        def duringTrigger():
            events.append('during')
        self.addTrigger('before', eventType, beforeTrigger)
        self.addTrigger('during', eventType, duringTrigger)
        self.assertEqual(events, [])
        reactor.fireSystemEvent(eventType)
        self.assertEqual(events, [])
        triggerDeferred.callback(None)
        self.assertEqual(events, ['during'])

    def test_multipleBeforeReturnDeferred(self):
        """
        If more than one trigger added to the C{'before'} phase of an event
        return L{Deferred}s, the C{'during'} phase should be delayed until they
        are all called back.
        """
        firstDeferred = Deferred()
        secondDeferred = Deferred()
        eventType = 'test'
        events = []

        def firstBeforeTrigger():
            return firstDeferred

        def secondBeforeTrigger():
            return secondDeferred

        def duringTrigger():
            events.append('during')
        self.addTrigger('before', eventType, firstBeforeTrigger)
        self.addTrigger('before', eventType, secondBeforeTrigger)
        self.addTrigger('during', eventType, duringTrigger)
        self.assertEqual(events, [])
        reactor.fireSystemEvent(eventType)
        self.assertEqual(events, [])
        firstDeferred.callback(None)
        self.assertEqual(events, [])
        secondDeferred.callback(None)
        self.assertEqual(events, ['during'])

    def test_subsequentBeforeTriggerFiresPriorBeforeDeferred(self):
        """
        If a trigger added to the C{'before'} phase of an event calls back a
        L{Deferred} returned by an earlier trigger in the C{'before'} phase of
        the same event, the remaining C{'before'} triggers for that event
        should be run and any further L{Deferred}s waited on before proceeding
        to the C{'during'} events.
        """
        eventType = 'test'
        events = []
        firstDeferred = Deferred()
        secondDeferred = Deferred()

        def firstBeforeTrigger():
            return firstDeferred

        def secondBeforeTrigger():
            firstDeferred.callback(None)

        def thirdBeforeTrigger():
            events.append('before')
            return secondDeferred

        def duringTrigger():
            events.append('during')
        self.addTrigger('before', eventType, firstBeforeTrigger)
        self.addTrigger('before', eventType, secondBeforeTrigger)
        self.addTrigger('before', eventType, thirdBeforeTrigger)
        self.addTrigger('during', eventType, duringTrigger)
        self.assertEqual(events, [])
        reactor.fireSystemEvent(eventType)
        self.assertEqual(events, ['before'])
        secondDeferred.callback(None)
        self.assertEqual(events, ['before', 'during'])

    def test_removeSystemEventTrigger(self):
        """
        A trigger removed with L{IReactorCore.removeSystemEventTrigger} should
        not be called when the event fires.
        """
        eventType = 'test'
        events = []

        def firstBeforeTrigger():
            events.append('first')

        def secondBeforeTrigger():
            events.append('second')
        self.addTrigger('before', eventType, firstBeforeTrigger)
        self.removeTrigger(self.addTrigger('before', eventType, secondBeforeTrigger))
        self.assertEqual(events, [])
        reactor.fireSystemEvent(eventType)
        self.assertEqual(events, ['first'])

    def test_removeNonExistentSystemEventTrigger(self):
        """
        Passing an object to L{IReactorCore.removeSystemEventTrigger} which was
        not returned by a previous call to
        L{IReactorCore.addSystemEventTrigger} or which has already been passed
        to C{removeSystemEventTrigger} should result in L{TypeError},
        L{KeyError}, or L{ValueError} being raised.
        """
        b = self.addTrigger('during', 'test', lambda: None)
        self.removeTrigger(b)
        self.assertRaises(TypeError, reactor.removeSystemEventTrigger, None)
        self.assertRaises(ValueError, reactor.removeSystemEventTrigger, b)
        self.assertRaises(KeyError, reactor.removeSystemEventTrigger, (b[0], ('xxx',) + b[1][1:]))

    def test_interactionBetweenDifferentEvents(self):
        """
        L{IReactorCore.fireSystemEvent} should behave the same way for a
        particular system event regardless of whether Deferreds are being
        waited on for a different system event.
        """
        events = []
        firstEvent = 'first-event'
        firstDeferred = Deferred()

        def beforeFirstEvent():
            events.append(('before', 'first'))
            return firstDeferred

        def afterFirstEvent():
            events.append(('after', 'first'))
        secondEvent = 'second-event'
        secondDeferred = Deferred()

        def beforeSecondEvent():
            events.append(('before', 'second'))
            return secondDeferred

        def afterSecondEvent():
            events.append(('after', 'second'))
        self.addTrigger('before', firstEvent, beforeFirstEvent)
        self.addTrigger('after', firstEvent, afterFirstEvent)
        self.addTrigger('before', secondEvent, beforeSecondEvent)
        self.addTrigger('after', secondEvent, afterSecondEvent)
        self.assertEqual(events, [])
        reactor.fireSystemEvent(firstEvent)
        self.assertEqual(events, [('before', 'first')])
        reactor.fireSystemEvent(secondEvent)
        self.assertEqual(events, [('before', 'first'), ('before', 'second')])
        firstDeferred.callback(None)
        self.assertEqual(events, [('before', 'first'), ('before', 'second'), ('after', 'first')])
        secondDeferred.callback(None)
        self.assertEqual(events, [('before', 'first'), ('before', 'second'), ('after', 'first'), ('after', 'second')])