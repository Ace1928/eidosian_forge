import unittest
from traits.api import (
class ArgCheckDecorator(ArgCheckBase):

    @on_trait_change('value')
    def arg_check0(self):
        self.calls += 1

    @on_trait_change('value')
    def arg_check1(self, new):
        self.calls += 1
        self.tc.assertEqual(new, self.value)

    @on_trait_change('value')
    def arg_check2(self, name, new):
        self.calls += 1
        self.tc.assertEqual(name, 'value')
        self.tc.assertEqual(new, self.value)

    @on_trait_change('value')
    def arg_check3(self, object, name, new):
        self.calls += 1
        self.tc.assertIs(object, self)
        self.tc.assertEqual(name, 'value')
        self.tc.assertEqual(new, self.value)

    @on_trait_change('value')
    def arg_check4(self, object, name, old, new):
        self.calls += 1
        self.tc.assertIs(object, self)
        self.tc.assertEqual(name, 'value')
        self.tc.assertEqual(old, self.value - 1)
        self.tc.assertEqual(new, self.value)