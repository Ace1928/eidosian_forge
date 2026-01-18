import collections.abc
import copy
import math
from unittest import mock
import ddt
from oslotest import base as test_base
import testscenarios
from oslo_utils import strutils
from oslo_utils import units
def test_xml(self):
    payload = '<adminPass>TL0EfN33</adminPass>'
    expected = '<adminPass>***</adminPass>'
    self.assertEqual(expected, strutils.mask_password(payload))
    payload = '<adminPass>\n                        TL0EfN33\n                     </adminPass>'
    expected = '<adminPass>***</adminPass>'
    self.assertEqual(expected, strutils.mask_password(payload))
    payload = '<admin_pass>TL0EfN33</admin_pass>'
    expected = '<admin_pass>***</admin_pass>'
    self.assertEqual(expected, strutils.mask_password(payload))
    payload = '<admin_pass>\n                        TL0EfN33\n                     </admin_pass>'
    expected = '<admin_pass>***</admin_pass>'
    self.assertEqual(expected, strutils.mask_password(payload))
    payload = '<admin_password>TL0EfN33</admin_password>'
    expected = '<admin_password>***</admin_password>'
    self.assertEqual(expected, strutils.mask_password(payload))
    payload = '<admin_password>\n                        TL0EfN33\n                     </admin_password>'
    expected = '<admin_password>***</admin_password>'
    self.assertEqual(expected, strutils.mask_password(payload))
    payload = '<password>TL0EfN33</password>'
    expected = '<password>***</password>'
    self.assertEqual(expected, strutils.mask_password(payload))
    payload = '<password>\n                        TL0EfN33\n                     </password>'
    expected = '<password>***</password>'
    self.assertEqual(expected, strutils.mask_password(payload))
    payload = '<Password1>TL0EfN33</Password1>'
    expected = '<Password1>***</Password1>'
    self.assertEqual(expected, strutils.mask_password(payload))