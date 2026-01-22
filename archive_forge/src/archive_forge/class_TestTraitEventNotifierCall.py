import unittest
from unittest import mock
import weakref
from traits.observation.exception_handling import (
from traits.observation.exceptions import NotifierNotFound
from traits.observation._trait_event_notifier import TraitEventNotifier
class TestTraitEventNotifierCall(unittest.TestCase):
    """ Test calling an instance of TraitEventNotifier. """

    def setUp(self):
        push_exception_handler(reraise_exceptions=True)
        self.addCleanup(pop_exception_handler)

    def tearDown(self):
        pass

    def test_init_and_call(self):
        handler = mock.Mock()

        def event_factory(*args, **kwargs):
            return 'Event'
        notifier = create_notifier(handler=handler, event_factory=event_factory)
        notifier(a=1, b=2)
        self.assertEqual(handler.call_count, 1)
        (args, _), = handler.call_args_list
        self.assertEqual(args, ('Event',))

    def test_alternative_dispatcher(self):
        events = []

        def dispatcher(handler, *args):
            event, = args
            events.append(event)

        def event_factory(*args, **kwargs):
            return 'Event'
        notifier = create_notifier(dispatcher=dispatcher, event_factory=event_factory)
        notifier(a=1, b=2)
        self.assertEqual(events, ['Event'])

    def test_prevent_event_is_used(self):

        def prevent_event(event):
            return True
        handler = mock.Mock()
        notifier = create_notifier(handler=handler, prevent_event=prevent_event)
        notifier(a=1, b=2)
        handler.assert_not_called()

    def test_init_check_handler_is_callable_early(self):
        not_a_callable = None
        with self.assertRaises(ValueError) as exception_cm:
            create_notifier(handler=not_a_callable)
        self.assertEqual(str(exception_cm.exception), 'handler must be a callable, got {!r}'.format(not_a_callable))