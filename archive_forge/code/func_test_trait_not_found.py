import unittest
from unittest import mock
from traits.api import Bool, HasTraits, Int, Instance
from traits.observation._named_trait_observer import (
from traits.observation._observer_graph import ObserverGraph
from traits.observation._testing import (
def test_trait_not_found(self):
    observer = create_observer(name='billy', optional=False)
    bar = ClassWithTwoValue()
    with self.assertRaises(ValueError) as e:
        next(observer.iter_observables(bar))
    self.assertEqual(str(e.exception), "Trait named 'billy' not found on {!r}.".format(bar))