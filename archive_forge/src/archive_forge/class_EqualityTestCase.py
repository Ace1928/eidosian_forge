import testscenarios
import oslo_messaging
from oslo_messaging.tests import utils as test_utils
class EqualityTestCase(test_utils.BaseTestCase):

    @classmethod
    def generate_scenarios(cls):
        attr = [('exchange', dict(attr='exchange')), ('topic', dict(attr='topic')), ('namespace', dict(attr='namespace')), ('version', dict(attr='version')), ('server', dict(attr='server')), ('fanout', dict(attr='fanout'))]
        a = [('a_notset', dict(a_value=_notset)), ('a_none', dict(a_value=None)), ('a_empty', dict(a_value='')), ('a_foo', dict(a_value='foo')), ('a_bar', dict(a_value='bar'))]
        b = [('b_notset', dict(b_value=_notset)), ('b_none', dict(b_value=None)), ('b_empty', dict(b_value='')), ('b_foo', dict(b_value='foo')), ('b_bar', dict(b_value='bar'))]
        cls.scenarios = testscenarios.multiply_scenarios(attr, a, b)
        for s in cls.scenarios:
            s[1]['equals'] = s[1]['a_value'] == s[1]['b_value']

    def test_equality(self):
        a_kwargs = {self.attr: self.a_value}
        b_kwargs = {self.attr: self.b_value}
        a = oslo_messaging.Target(**a_kwargs)
        b = oslo_messaging.Target(**b_kwargs)
        if self.equals:
            self.assertEqual(a, b)
            self.assertFalse(a != b)
        else:
            self.assertNotEqual(a, b)
            self.assertFalse(a == b)