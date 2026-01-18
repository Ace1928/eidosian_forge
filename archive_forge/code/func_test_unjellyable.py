import datetime
import decimal
from twisted.internet.testing import StringTransport
from twisted.spread import banana, jelly, pb
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
def test_unjellyable(self):
    """
        Test that if Unjellyable is used to deserialize a jellied object,
        state comes out right.
        """

    class JellyableTestClass(jelly.Jellyable):
        pass
    jelly.setUnjellyableForClass(JellyableTestClass, jelly.Unjellyable)
    input = JellyableTestClass()
    input.attribute = 'value'
    output = jelly.unjelly(jelly.jelly(input))
    self.assertEqual(output.attribute, 'value')
    self.assertIsInstance(output, jelly.Unjellyable)