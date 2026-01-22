import collections.abc
import copy
import math
from unittest import mock
import ddt
from oslotest import base as test_base
import testscenarios
from oslo_utils import strutils
from oslo_utils import units
class StringToBytesTest(test_base.BaseTestCase):
    _unit_system = [('si', dict(unit_system='SI')), ('iec', dict(unit_system='IEC')), ('mixed', dict(unit_system='mixed')), ('invalid_unit_system', dict(unit_system='KKK', assert_error=True))]
    _sign = [('no_sign', dict(sign='')), ('positive', dict(sign='+')), ('negative', dict(sign='-')), ('invalid_sign', dict(sign='~', assert_error=True))]
    _magnitude = [('integer', dict(magnitude='79')), ('decimal', dict(magnitude='7.9')), ('decimal_point_start', dict(magnitude='.9')), ('decimal_point_end', dict(magnitude='79.', assert_error=True)), ('invalid_literal', dict(magnitude='7.9.9', assert_error=True)), ('garbage_value', dict(magnitude='asdf', assert_error=True))]
    _unit_prefix = [('no_unit_prefix', dict(unit_prefix='')), ('k', dict(unit_prefix='k')), ('K', dict(unit_prefix='K')), ('M', dict(unit_prefix='M')), ('G', dict(unit_prefix='G')), ('T', dict(unit_prefix='T')), ('P', dict(unit_prefix='P')), ('E', dict(unit_prefix='E')), ('Z', dict(unit_prefix='Z')), ('Y', dict(unit_prefix='Y')), ('R', dict(unit_prefix='R')), ('Q', dict(unit_prefix='Q')), ('Ki', dict(unit_prefix='Ki')), ('Mi', dict(unit_prefix='Mi')), ('Gi', dict(unit_prefix='Gi')), ('Ti', dict(unit_prefix='Ti')), ('Pi', dict(unit_prefix='Pi')), ('Ei', dict(unit_prefix='Ei')), ('Zi', dict(unit_prefix='Zi')), ('Yi', dict(unit_prefix='Yi')), ('Ri', dict(unit_prefix='Ri')), ('Qi', dict(unit_prefix='Qi')), ('invalid_unit_prefix', dict(unit_prefix='B', assert_error=True))]
    _unit_suffix = [('b', dict(unit_suffix='b')), ('bit', dict(unit_suffix='bit')), ('B', dict(unit_suffix='B')), ('invalid_unit_suffix', dict(unit_suffix='Kg', assert_error=True))]
    _return_int = [('return_dec', dict(return_int=False)), ('return_int', dict(return_int=True))]

    @classmethod
    def generate_scenarios(cls):
        cls.scenarios = testscenarios.multiply_scenarios(cls._unit_system, cls._sign, cls._magnitude, cls._unit_prefix, cls._unit_suffix, cls._return_int)

    def test_string_to_bytes(self):

        def _get_quantity(sign, magnitude, unit_suffix):
            res = float('%s%s' % (sign, magnitude))
            if unit_suffix in ['b', 'bit']:
                res /= 8
            return res

        def _get_constant(unit_prefix, unit_system):
            if not unit_prefix:
                return 1
            elif unit_system == 'SI':
                res = getattr(units, unit_prefix)
            elif unit_system == 'IEC':
                if unit_prefix.endswith('i'):
                    res = getattr(units, unit_prefix)
                else:
                    res = getattr(units, '%si' % unit_prefix)
            elif unit_system == 'mixed':
                if unit_prefix == 'K':
                    unit_prefix = 'k'
                res = getattr(units, unit_prefix)
            return res
        text = ''.join([self.sign, self.magnitude, self.unit_prefix, self.unit_suffix])
        err_si = self.unit_system == 'SI' and (self.unit_prefix == 'K' or self.unit_prefix.endswith('i'))
        err_iec = self.unit_system == 'IEC' and self.unit_prefix == 'k'
        if getattr(self, 'assert_error', False) or err_si or err_iec:
            self.assertRaises(ValueError, strutils.string_to_bytes, text, unit_system=self.unit_system, return_int=self.return_int)
            return
        quantity = _get_quantity(self.sign, self.magnitude, self.unit_suffix)
        constant = _get_constant(self.unit_prefix, self.unit_system)
        expected = quantity * constant
        actual = strutils.string_to_bytes(text, unit_system=self.unit_system, return_int=self.return_int)
        if self.return_int:
            self.assertEqual(actual, int(math.ceil(expected)))
        else:
            self.assertAlmostEqual(actual, expected)