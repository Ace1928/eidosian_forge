import pickle
import unittest
from unittest import mock
from traits.api import HasTraits, Set, Str
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_set_object import TraitSet, TraitSetEvent
from traits.trait_types import _validate_int
def test_pickle_with_notifier(self):
    foo = Foo(values={1, 2, 3})
    foo.values.notifiers.append(notifier)
    protocols = range(pickle.HIGHEST_PROTOCOL + 1)
    for protocol in protocols:
        with self.subTest(protocol=protocol):
            serialized = pickle.dumps(foo.values, protocol=protocol)
            deserialized = pickle.loads(serialized)
            self.assertEqual(deserialized.notifiers, [deserialized.notifier])