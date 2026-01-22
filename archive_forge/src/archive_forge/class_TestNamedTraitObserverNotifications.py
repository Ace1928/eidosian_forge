import unittest
from unittest import mock
from traits.api import Bool, HasTraits, Int, Instance
from traits.observation._named_trait_observer import (
from traits.observation._observer_graph import ObserverGraph
from traits.observation._testing import (
class TestNamedTraitObserverNotifications(unittest.TestCase):
    """ Test integration with observe and HasTraits
    to get notifications.
    """

    def test_notifier_extended_trait_change(self):
        foo = ClassWithInstance()
        graph = create_graph(create_observer(name='instance', notify=True), create_observer(name='value1', notify=True))
        handler = mock.Mock()
        call_add_or_remove_notifiers(object=foo, graph=graph, handler=handler)
        self.assertIsNone(foo.instance)
        foo.instance = ClassWithTwoValue()
        ((event,), _), = handler.call_args_list
        self.assertEqual(event.object, foo)
        self.assertEqual(event.name, 'instance')
        self.assertEqual(event.old, None)
        self.assertEqual(event.new, foo.instance)
        handler.reset_mock()
        foo.instance.value1 += 1
        ((event,), _), = handler.call_args_list
        self.assertEqual(event.object, foo.instance)
        self.assertEqual(event.name, 'value1')
        self.assertEqual(event.old, 0)
        self.assertEqual(event.new, 1)

    def test_maintain_notifier_change_to_new_value(self):
        foo = ClassWithInstance()
        graph = create_graph(create_observer(name='instance', notify=True), create_observer(name='value1', notify=True))
        handler = mock.Mock()
        call_add_or_remove_notifiers(object=foo, graph=graph, handler=handler)
        foo.instance = ClassWithTwoValue()
        foo.instance.value1 += 1
        self.assertEqual(handler.call_count, 2)
        old_instance = foo.instance
        foo.instance = ClassWithTwoValue()
        handler.reset_mock()
        old_instance.value1 += 1
        self.assertEqual(handler.call_count, 0)
        foo.instance.value1 += 1
        self.assertEqual(handler.call_count, 1)

    def test_maintain_notifier_change_to_none(self):

        class UnassumingObserver(DummyObserver):

            def iter_observables(self, object):
                if object is None:
                    raise ValueError('This observer cannot handle None.')
                yield from ()
        foo = ClassWithInstance()
        graph = create_graph(create_observer(name='instance', notify=True), UnassumingObserver())
        handler = mock.Mock()
        call_add_or_remove_notifiers(object=foo, graph=graph, handler=handler)
        foo.instance = ClassWithTwoValue()
        try:
            foo.instance = None
        except Exception:
            self.fail('Setting instance back to None should not fail.')

    def test_maintain_notifier_for_default(self):
        foo = ClassWithDefault()
        graph = create_graph(create_observer(name='instance', notify=True), create_observer(name='value1', notify=True))
        handler = mock.Mock()
        call_add_or_remove_notifiers(object=foo, graph=graph, handler=handler)
        self.assertNotIn('instance', foo.__dict__)
        foo.instance
        self.assertEqual(handler.call_count, 0)
        foo.instance.value1 += 1
        self.assertEqual(handler.call_count, 1)

    def test_get_maintainer_excuse_old_value_with_no_notifiers(self):
        foo = ClassWithDefault()
        graph = create_graph(create_observer(name='instance', notify=True), create_observer(name='value1', notify=True))
        handler = mock.Mock()
        call_add_or_remove_notifiers(object=foo, graph=graph, handler=handler)
        try:
            foo.instance = ClassWithTwoValue()
        except Exception:
            self.fail('Reassigning the instance value should not fail.')