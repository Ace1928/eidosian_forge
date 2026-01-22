import unittest
from unittest import mock
from traits.api import Bool, HasTraits, Int, Instance
from traits.observation._named_trait_observer import (
from traits.observation._observer_graph import ObserverGraph
from traits.observation._testing import (
class TestNamedTraitObserverNextObjects(unittest.TestCase):
    """ Tests for NamedTraitObserver.iter_objects for the downstream
    observers.
    """

    def test_iter_objects(self):
        observer = create_observer(name='instance')
        foo = ClassWithInstance(instance=ClassWithTwoValue())
        actual = list(observer.iter_objects(foo))
        self.assertEqual(actual, [foo.instance])

    def test_iter_objects_raises_if_trait_not_found(self):
        observer = create_observer(name='sally', optional=False)
        foo = ClassWithInstance()
        with self.assertRaises(ValueError) as e:
            next(observer.iter_objects(foo))
        self.assertEqual(str(e.exception), 'Trait named {!r} not found on {!r}.'.format('sally', foo))

    def test_trait_not_found_skip_as_optional(self):
        observer = create_observer(name='sally', optional=True)
        foo = ClassWithInstance()
        actual = list(observer.iter_objects(foo))
        self.assertEqual(actual, [])

    def test_iter_objects_no_side_effect_on_default_initializer(self):
        observer = create_observer(name='instance')
        foo = ClassWithDefault()
        actual = list(observer.iter_objects(foo))
        self.assertEqual(actual, [])
        self.assertNotIn('instance', foo.__dict__)
        self.assertFalse(foo.instance_default_calculated, 'Unexpected side-effect on the default initializer.')