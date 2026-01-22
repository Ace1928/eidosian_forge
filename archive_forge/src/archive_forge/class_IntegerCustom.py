from ._IntegerNative import IntegerNative
from Cryptodome.Util.number import long_to_bytes, bytes_to_long
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib,
from Cryptodome.Random.random import getrandbits
class IntegerCustom(IntegerNative):

    @staticmethod
    def from_bytes(byte_string, byteorder='big'):
        if byteorder == 'big':
            pass
        elif byteorder == 'little':
            byte_string = bytearray(byte_string)
            byte_string.reverse()
        else:
            raise ValueError('Incorrect byteorder')
        return IntegerCustom(bytes_to_long(byte_string))

    def inplace_pow(self, exponent, modulus=None):
        exp_value = int(exponent)
        if exp_value < 0:
            raise ValueError('Exponent must not be negative')
        if modulus is None:
            self._value = pow(self._value, exp_value)
            return self
        mod_value = int(modulus)
        if mod_value < 0:
            raise ValueError('Modulus must be positive')
        if mod_value == 0:
            raise ZeroDivisionError('Modulus cannot be zero')
        if mod_value & 1 == 0:
            self._value = pow(self._value, exp_value, mod_value)
            return self
        if self._value >= mod_value:
            self._value %= mod_value
        max_len = len(long_to_bytes(max(self._value, exp_value, mod_value)))
        base_b = long_to_bytes(self._value, max_len)
        exp_b = long_to_bytes(exp_value, max_len)
        modulus_b = long_to_bytes(mod_value, max_len)
        out = create_string_buffer(max_len)
        error = _raw_montgomery.monty_pow(out, base_b, exp_b, modulus_b, c_size_t(max_len), c_ulonglong(getrandbits(64)))
        if error:
            raise ValueError('monty_pow failed with error: %d' % error)
        result = bytes_to_long(get_raw_buffer(out))
        self._value = result
        return self

    @staticmethod
    def _mult_modulo_bytes(term1, term2, modulus):
        mod_value = int(modulus)
        if mod_value < 0:
            raise ValueError('Modulus must be positive')
        if mod_value == 0:
            raise ZeroDivisionError('Modulus cannot be zero')
        if mod_value & 1 == 0:
            raise ValueError('Odd modulus is required')
        if term1 >= mod_value or term1 < 0:
            term1 %= mod_value
        if term2 >= mod_value or term2 < 0:
            term2 %= mod_value
        modulus_b = long_to_bytes(mod_value)
        numbers_len = len(modulus_b)
        term1_b = long_to_bytes(term1, numbers_len)
        term2_b = long_to_bytes(term2, numbers_len)
        out = create_string_buffer(numbers_len)
        error = _raw_montgomery.monty_multiply(out, term1_b, term2_b, modulus_b, c_size_t(numbers_len))
        if error:
            raise ValueError('monty_multiply failed with error: %d' % error)
        return get_raw_buffer(out)