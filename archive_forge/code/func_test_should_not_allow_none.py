import unittest
from traits.api import CList, HasTraits, Instance, Int, List, Str, TraitError
def test_should_not_allow_none(self):
    f = Foo(l=['initial', 'value'])
    try:
        f.l = None
        self.fail('None assigned to List trait.')
    except TraitError:
        pass