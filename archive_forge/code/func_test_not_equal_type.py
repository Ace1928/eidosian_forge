import unittest
from unittest import mock
from traits.api import Bool, HasTraits, Int, Instance
from traits.observation._named_trait_observer import (
from traits.observation._observer_graph import ObserverGraph
from traits.observation._testing import (
def test_not_equal_type(self):
    observer = NamedTraitObserver(name='foo', notify=True, optional=True)
    imposter = mock.Mock()
    imposter.name = 'foo'
    imposter.notify = True
    imposter.optional = True
    self.assertNotEqual(observer, imposter)