import unittest
from traits import trait_notifiers
from traits.api import Float, HasTraits, List
 Tests for the extended notifiers.

The "extended notifiers" are set up internally when using extended traits, to
add/remove traits listeners when one of the intermediate traits changes.

For example, in a listener for the extended trait `a.b`, we need to add/remove
listeners to `a:b` when `a` changes.
