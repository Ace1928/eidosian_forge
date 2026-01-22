import unittest
from unittest import mock
from traits.api import Bool, HasTraits, Int, Instance
from traits.observation._named_trait_observer import (
from traits.observation._observer_graph import ObserverGraph
from traits.observation._testing import (
class TestNamedTraitObserverIterObservable(unittest.TestCase):
    """ Tests for NamedTraitObserver.iter_observables """

    def test_ordinary_has_traits(self):
        observer = create_observer(name='value1', optional=False)
        foo = ClassWithTwoValue()
        actual = list(observer.iter_observables(foo))
        self.assertEqual(actual, [foo._trait('value1', 2)])

    def test_trait_not_found(self):
        observer = create_observer(name='billy', optional=False)
        bar = ClassWithTwoValue()
        with self.assertRaises(ValueError) as e:
            next(observer.iter_observables(bar))
        self.assertEqual(str(e.exception), "Trait named 'billy' not found on {!r}.".format(bar))

    def test_trait_not_found_skip_as_optional(self):
        observer = create_observer(name='billy', optional=True)
        bar = ClassWithTwoValue()
        actual = list(observer.iter_observables(bar))
        self.assertEqual(actual, [])