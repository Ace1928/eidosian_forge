import calendar
import datetime
import decimal
import json
import locale
import math
import re
import time
import dateutil
import numpy as np
import pytest
import pytz
import pandas._libs.json as ujson
from pandas.compat import IS64
from pandas import (
import pandas._testing as tm
class TestUltraJSONTests:

    @pytest.mark.skipif(not IS64, reason='not compliant on 32-bit, xref #15865')
    def test_encode_decimal(self):
        sut = decimal.Decimal('1337.1337')
        encoded = ujson.ujson_dumps(sut, double_precision=15)
        decoded = ujson.ujson_loads(encoded)
        assert decoded == 1337.1337
        sut = decimal.Decimal('0.95')
        encoded = ujson.ujson_dumps(sut, double_precision=1)
        assert encoded == '1.0'
        decoded = ujson.ujson_loads(encoded)
        assert decoded == 1.0
        sut = decimal.Decimal('0.94')
        encoded = ujson.ujson_dumps(sut, double_precision=1)
        assert encoded == '0.9'
        decoded = ujson.ujson_loads(encoded)
        assert decoded == 0.9
        sut = decimal.Decimal('1.95')
        encoded = ujson.ujson_dumps(sut, double_precision=1)
        assert encoded == '2.0'
        decoded = ujson.ujson_loads(encoded)
        assert decoded == 2.0
        sut = decimal.Decimal('-1.95')
        encoded = ujson.ujson_dumps(sut, double_precision=1)
        assert encoded == '-2.0'
        decoded = ujson.ujson_loads(encoded)
        assert decoded == -2.0
        sut = decimal.Decimal('0.995')
        encoded = ujson.ujson_dumps(sut, double_precision=2)
        assert encoded == '1.0'
        decoded = ujson.ujson_loads(encoded)
        assert decoded == 1.0
        sut = decimal.Decimal('0.9995')
        encoded = ujson.ujson_dumps(sut, double_precision=3)
        assert encoded == '1.0'
        decoded = ujson.ujson_loads(encoded)
        assert decoded == 1.0
        sut = decimal.Decimal('0.99999999999999944')
        encoded = ujson.ujson_dumps(sut, double_precision=15)
        assert encoded == '1.0'
        decoded = ujson.ujson_loads(encoded)
        assert decoded == 1.0

    @pytest.mark.parametrize('ensure_ascii', [True, False])
    def test_encode_string_conversion(self, ensure_ascii):
        string_input = 'A string \\ / \x08 \x0c \n \r \t </script> &'
        not_html_encoded = '"A string \\\\ \\/ \\b \\f \\n \\r \\t <\\/script> &"'
        html_encoded = '"A string \\\\ \\/ \\b \\f \\n \\r \\t \\u003c\\/script\\u003e \\u0026"'

        def helper(expected_output, **encode_kwargs):
            output = ujson.ujson_dumps(string_input, ensure_ascii=ensure_ascii, **encode_kwargs)
            assert output == expected_output
            assert string_input == json.loads(output)
            assert string_input == ujson.ujson_loads(output)
        helper(not_html_encoded)
        helper(not_html_encoded, encode_html_chars=False)
        helper(html_encoded, encode_html_chars=True)

    @pytest.mark.parametrize('long_number', [-4342969734183514, -12345678901234.568, -528656961.4399388])
    def test_double_long_numbers(self, long_number):
        sut = {'a': long_number}
        encoded = ujson.ujson_dumps(sut, double_precision=15)
        decoded = ujson.ujson_loads(encoded)
        assert sut == decoded

    def test_encode_non_c_locale(self):
        lc_category = locale.LC_NUMERIC
        for new_locale in ('it_IT.UTF-8', 'Italian_Italy'):
            if tm.can_set_locale(new_locale, lc_category):
                with tm.set_locale(new_locale, lc_category):
                    assert ujson.ujson_loads(ujson.ujson_dumps(4.78e+60)) == 4.78e+60
                    assert ujson.ujson_loads('4.78', precise_float=True) == 4.78
                break

    def test_decimal_decode_test_precise(self):
        sut = {'a': 4.56}
        encoded = ujson.ujson_dumps(sut)
        decoded = ujson.ujson_loads(encoded, precise_float=True)
        assert sut == decoded

    def test_encode_double_tiny_exponential(self):
        num = 1e-40
        assert num == ujson.ujson_loads(ujson.ujson_dumps(num))
        num = 1e-100
        assert num == ujson.ujson_loads(ujson.ujson_dumps(num))
        num = -1e-45
        assert num == ujson.ujson_loads(ujson.ujson_dumps(num))
        num = -1e-145
        assert np.allclose(num, ujson.ujson_loads(ujson.ujson_dumps(num)))

    @pytest.mark.parametrize('unicode_key', ['key1', 'بن'])
    def test_encode_dict_with_unicode_keys(self, unicode_key):
        unicode_dict = {unicode_key: 'value1'}
        assert unicode_dict == ujson.ujson_loads(ujson.ujson_dumps(unicode_dict))

    @pytest.mark.parametrize('double_input', [math.pi, -math.pi])
    def test_encode_double_conversion(self, double_input):
        output = ujson.ujson_dumps(double_input)
        assert round(double_input, 5) == round(json.loads(output), 5)
        assert round(double_input, 5) == round(ujson.ujson_loads(output), 5)

    def test_encode_with_decimal(self):
        decimal_input = 1.0
        output = ujson.ujson_dumps(decimal_input)
        assert output == '1.0'

    def test_encode_array_of_nested_arrays(self):
        nested_input = [[[[]]]] * 20
        output = ujson.ujson_dumps(nested_input)
        assert nested_input == json.loads(output)
        assert nested_input == ujson.ujson_loads(output)

    def test_encode_array_of_doubles(self):
        doubles_input = [31337.31337, 31337.31337, 31337.31337, 31337.31337] * 10
        output = ujson.ujson_dumps(doubles_input)
        assert doubles_input == json.loads(output)
        assert doubles_input == ujson.ujson_loads(output)

    def test_double_precision(self):
        double_input = 30.012345678901234
        output = ujson.ujson_dumps(double_input, double_precision=15)
        assert double_input == json.loads(output)
        assert double_input == ujson.ujson_loads(output)
        for double_precision in (3, 9):
            output = ujson.ujson_dumps(double_input, double_precision=double_precision)
            rounded_input = round(double_input, double_precision)
            assert rounded_input == json.loads(output)
            assert rounded_input == ujson.ujson_loads(output)

    @pytest.mark.parametrize('invalid_val', [20, -1, '9', None])
    def test_invalid_double_precision(self, invalid_val):
        double_input = 30.123456789012344
        expected_exception = ValueError if isinstance(invalid_val, int) else TypeError
        msg = "Invalid value '.*' for option 'double_precision', max is '15'|an integer is required \\(got type |object cannot be interpreted as an integer"
        with pytest.raises(expected_exception, match=msg):
            ujson.ujson_dumps(double_input, double_precision=invalid_val)

    def test_encode_string_conversion2(self):
        string_input = 'A string \\ / \x08 \x0c \n \r \t'
        output = ujson.ujson_dumps(string_input)
        assert string_input == json.loads(output)
        assert string_input == ujson.ujson_loads(output)
        assert output == '"A string \\\\ \\/ \\b \\f \\n \\r \\t"'

    @pytest.mark.parametrize('unicode_input', ['Räksmörgås اسامة بن محمد بن عوض بن لادن', 'æ\x97¥Ñ\x88'])
    def test_encode_unicode_conversion(self, unicode_input):
        enc = ujson.ujson_dumps(unicode_input)
        dec = ujson.ujson_loads(enc)
        assert enc == json.dumps(unicode_input)
        assert dec == json.loads(enc)

    def test_encode_control_escaping(self):
        escaped_input = '\x19'
        enc = ujson.ujson_dumps(escaped_input)
        dec = ujson.ujson_loads(enc)
        assert escaped_input == dec
        assert enc == json.dumps(escaped_input)

    def test_encode_unicode_surrogate_pair(self):
        surrogate_input = 'ð\x90\x8d\x86'
        enc = ujson.ujson_dumps(surrogate_input)
        dec = ujson.ujson_loads(enc)
        assert enc == json.dumps(surrogate_input)
        assert dec == json.loads(enc)

    def test_encode_unicode_4bytes_utf8(self):
        four_bytes_input = 'ð\x91\x80°TRAILINGNORMAL'
        enc = ujson.ujson_dumps(four_bytes_input)
        dec = ujson.ujson_loads(enc)
        assert enc == json.dumps(four_bytes_input)
        assert dec == json.loads(enc)

    def test_encode_unicode_4bytes_utf8highest(self):
        four_bytes_input = 'ó¿¿¿TRAILINGNORMAL'
        enc = ujson.ujson_dumps(four_bytes_input)
        dec = ujson.ujson_loads(enc)
        assert enc == json.dumps(four_bytes_input)
        assert dec == json.loads(enc)

    def test_encode_unicode_error(self):
        string = "'\udac0'"
        msg = "'utf-8' codec can't encode character '\\\\udac0' in position 1: surrogates not allowed"
        with pytest.raises(UnicodeEncodeError, match=msg):
            ujson.ujson_dumps([string])

    def test_encode_array_in_array(self):
        arr_in_arr_input = [[[[]]]]
        output = ujson.ujson_dumps(arr_in_arr_input)
        assert arr_in_arr_input == json.loads(output)
        assert output == json.dumps(arr_in_arr_input)
        assert arr_in_arr_input == ujson.ujson_loads(output)

    @pytest.mark.parametrize('num_input', [31337, -31337, -9223372036854775808])
    def test_encode_num_conversion(self, num_input):
        output = ujson.ujson_dumps(num_input)
        assert num_input == json.loads(output)
        assert output == json.dumps(num_input)
        assert num_input == ujson.ujson_loads(output)

    def test_encode_list_conversion(self):
        list_input = [1, 2, 3, 4]
        output = ujson.ujson_dumps(list_input)
        assert list_input == json.loads(output)
        assert list_input == ujson.ujson_loads(output)

    def test_encode_dict_conversion(self):
        dict_input = {'k1': 1, 'k2': 2, 'k3': 3, 'k4': 4}
        output = ujson.ujson_dumps(dict_input)
        assert dict_input == json.loads(output)
        assert dict_input == ujson.ujson_loads(output)

    @pytest.mark.parametrize('builtin_value', [None, True, False])
    def test_encode_builtin_values_conversion(self, builtin_value):
        output = ujson.ujson_dumps(builtin_value)
        assert builtin_value == json.loads(output)
        assert output == json.dumps(builtin_value)
        assert builtin_value == ujson.ujson_loads(output)

    def test_encode_datetime_conversion(self):
        datetime_input = datetime.datetime.fromtimestamp(time.time())
        output = ujson.ujson_dumps(datetime_input, date_unit='s')
        expected = calendar.timegm(datetime_input.utctimetuple())
        assert int(expected) == json.loads(output)
        assert int(expected) == ujson.ujson_loads(output)

    def test_encode_date_conversion(self):
        date_input = datetime.date.fromtimestamp(time.time())
        output = ujson.ujson_dumps(date_input, date_unit='s')
        tup = (date_input.year, date_input.month, date_input.day, 0, 0, 0)
        expected = calendar.timegm(tup)
        assert int(expected) == json.loads(output)
        assert int(expected) == ujson.ujson_loads(output)

    @pytest.mark.parametrize('test', [datetime.time(), datetime.time(1, 2, 3), datetime.time(10, 12, 15, 343243)])
    def test_encode_time_conversion_basic(self, test):
        output = ujson.ujson_dumps(test)
        expected = f'"{test.isoformat()}"'
        assert expected == output

    def test_encode_time_conversion_pytz(self):
        test = datetime.time(10, 12, 15, 343243, pytz.utc)
        output = ujson.ujson_dumps(test)
        expected = f'"{test.isoformat()}"'
        assert expected == output

    def test_encode_time_conversion_dateutil(self):
        test = datetime.time(10, 12, 15, 343243, dateutil.tz.tzutc())
        output = ujson.ujson_dumps(test)
        expected = f'"{test.isoformat()}"'
        assert expected == output

    @pytest.mark.parametrize('decoded_input', [NaT, np.datetime64('NaT'), np.nan, np.inf, -np.inf])
    def test_encode_as_null(self, decoded_input):
        assert ujson.ujson_dumps(decoded_input) == 'null', 'Expected null'

    def test_datetime_units(self):
        val = datetime.datetime(2013, 8, 17, 21, 17, 12, 215504)
        stamp = Timestamp(val).as_unit('ns')
        roundtrip = ujson.ujson_loads(ujson.ujson_dumps(val, date_unit='s'))
        assert roundtrip == stamp._value // 10 ** 9
        roundtrip = ujson.ujson_loads(ujson.ujson_dumps(val, date_unit='ms'))
        assert roundtrip == stamp._value // 10 ** 6
        roundtrip = ujson.ujson_loads(ujson.ujson_dumps(val, date_unit='us'))
        assert roundtrip == stamp._value // 10 ** 3
        roundtrip = ujson.ujson_loads(ujson.ujson_dumps(val, date_unit='ns'))
        assert roundtrip == stamp._value
        msg = "Invalid value 'foo' for option 'date_unit'"
        with pytest.raises(ValueError, match=msg):
            ujson.ujson_dumps(val, date_unit='foo')

    def test_encode_to_utf8(self):
        unencoded = 'æ\x97¥Ñ\x88'
        enc = ujson.ujson_dumps(unencoded, ensure_ascii=False)
        dec = ujson.ujson_loads(enc)
        assert enc == json.dumps(unencoded, ensure_ascii=False)
        assert dec == json.loads(enc)

    def test_decode_from_unicode(self):
        unicode_input = '{"obj": 31337}'
        dec1 = ujson.ujson_loads(unicode_input)
        dec2 = ujson.ujson_loads(str(unicode_input))
        assert dec1 == dec2

    def test_encode_recursion_max(self):

        class O2:
            member = 0

        class O1:
            member = 0
        decoded_input = O1()
        decoded_input.member = O2()
        decoded_input.member.member = decoded_input
        with pytest.raises(OverflowError, match='Maximum recursion level reached'):
            ujson.ujson_dumps(decoded_input)

    def test_decode_jibberish(self):
        jibberish = 'fdsa sda v9sa fdsa'
        msg = "Unexpected character found when decoding 'false'"
        with pytest.raises(ValueError, match=msg):
            ujson.ujson_loads(jibberish)

    @pytest.mark.parametrize('broken_json', ['[', '{', ']', '}'])
    def test_decode_broken_json(self, broken_json):
        msg = 'Expected object or value'
        with pytest.raises(ValueError, match=msg):
            ujson.ujson_loads(broken_json)

    @pytest.mark.parametrize('too_big_char', ['[', '{'])
    def test_decode_depth_too_big(self, too_big_char):
        with pytest.raises(ValueError, match='Reached object decoding depth limit'):
            ujson.ujson_loads(too_big_char * (1024 * 1024))

    @pytest.mark.parametrize('bad_string', ['"TESTING', '"TESTING\\"', 'tru', 'fa', 'n'])
    def test_decode_bad_string(self, bad_string):
        msg = 'Unexpected character found when decoding|Unmatched \'\'"\' when when decoding \'string\''
        with pytest.raises(ValueError, match=msg):
            ujson.ujson_loads(bad_string)

    @pytest.mark.parametrize('broken_json, err_msg', [('{{1337:""}}', "Key name of object must be 'string' when decoding 'object'"), ('{{"key":"}', 'Unmatched \'\'"\' when when decoding \'string\''), ('[[[true', 'Unexpected character found when decoding array value (2)')])
    def test_decode_broken_json_leak(self, broken_json, err_msg):
        for _ in range(1000):
            with pytest.raises(ValueError, match=re.escape(err_msg)):
                ujson.ujson_loads(broken_json)

    @pytest.mark.parametrize('invalid_dict', ['{{{{31337}}}}', '{{{{"key":}}}}', '{{{{"key"}}}}'])
    def test_decode_invalid_dict(self, invalid_dict):
        msg = "Key name of object must be 'string' when decoding 'object'|No ':' found when decoding object value|Expected object or value"
        with pytest.raises(ValueError, match=msg):
            ujson.ujson_loads(invalid_dict)

    @pytest.mark.parametrize('numeric_int_as_str', ['31337', '-31337'])
    def test_decode_numeric_int(self, numeric_int_as_str):
        assert int(numeric_int_as_str) == ujson.ujson_loads(numeric_int_as_str)

    def test_encode_null_character(self):
        wrapped_input = '31337 \x00 1337'
        output = ujson.ujson_dumps(wrapped_input)
        assert wrapped_input == json.loads(output)
        assert output == json.dumps(wrapped_input)
        assert wrapped_input == ujson.ujson_loads(output)
        alone_input = '\x00'
        output = ujson.ujson_dumps(alone_input)
        assert alone_input == json.loads(output)
        assert output == json.dumps(alone_input)
        assert alone_input == ujson.ujson_loads(output)
        assert '"  \\u0000\\r\\n "' == ujson.ujson_dumps('  \x00\r\n ')

    def test_decode_null_character(self):
        wrapped_input = '"31337 \\u0000 31337"'
        assert ujson.ujson_loads(wrapped_input) == json.loads(wrapped_input)

    def test_encode_list_long_conversion(self):
        long_input = [9223372036854775807, 9223372036854775807, 9223372036854775807, 9223372036854775807, 9223372036854775807, 9223372036854775807]
        output = ujson.ujson_dumps(long_input)
        assert long_input == json.loads(output)
        assert long_input == ujson.ujson_loads(output)

    @pytest.mark.parametrize('long_input', [9223372036854775807, 18446744073709551615])
    def test_encode_long_conversion(self, long_input):
        output = ujson.ujson_dumps(long_input)
        assert long_input == json.loads(output)
        assert output == json.dumps(long_input)
        assert long_input == ujson.ujson_loads(output)

    @pytest.mark.parametrize('bigNum', [2 ** 64, -2 ** 63 - 1])
    def test_dumps_ints_larger_than_maxsize(self, bigNum):
        encoding = ujson.ujson_dumps(bigNum)
        assert str(bigNum) == encoding
        with pytest.raises(ValueError, match='Value is too big|Value is too small'):
            assert ujson.ujson_loads(encoding) == bigNum

    @pytest.mark.parametrize('int_exp', ['1337E40', '1.337E40', '1337E+9', '1.337e+40', '1.337E-4'])
    def test_decode_numeric_int_exp(self, int_exp):
        assert ujson.ujson_loads(int_exp) == json.loads(int_exp)

    def test_loads_non_str_bytes_raises(self):
        msg = "a bytes-like object is required, not 'NoneType'"
        with pytest.raises(TypeError, match=msg):
            ujson.ujson_loads(None)

    @pytest.mark.parametrize('val', [3590016419, 2 ** 31, 2 ** 32, 2 ** 32 - 1])
    def test_decode_number_with_32bit_sign_bit(self, val):
        doc = f'{{"id": {val}}}'
        assert ujson.ujson_loads(doc)['id'] == val

    def test_encode_big_escape(self):
        for _ in range(10):
            base = 'å'.encode()
            escape_input = base * 1024 * 1024 * 2
            ujson.ujson_dumps(escape_input)

    def test_decode_big_escape(self):
        for _ in range(10):
            base = 'å'.encode()
            quote = b'"'
            escape_input = quote + base * 1024 * 1024 * 2 + quote
            ujson.ujson_loads(escape_input)

    def test_to_dict(self):
        d = {'key': 31337}

        class DictTest:

            def toDict(self):
                return d
        o = DictTest()
        output = ujson.ujson_dumps(o)
        dec = ujson.ujson_loads(output)
        assert dec == d

    def test_default_handler(self):

        class _TestObject:

            def __init__(self, val) -> None:
                self.val = val

            @property
            def recursive_attr(self):
                return _TestObject('recursive_attr')

            def __str__(self) -> str:
                return str(self.val)
        msg = 'Maximum recursion level reached'
        with pytest.raises(OverflowError, match=msg):
            ujson.ujson_dumps(_TestObject('foo'))
        assert '"foo"' == ujson.ujson_dumps(_TestObject('foo'), default_handler=str)

        def my_handler(_):
            return 'foobar'
        assert '"foobar"' == ujson.ujson_dumps(_TestObject('foo'), default_handler=my_handler)

        def my_handler_raises(_):
            raise TypeError('I raise for anything')
        with pytest.raises(TypeError, match='I raise for anything'):
            ujson.ujson_dumps(_TestObject('foo'), default_handler=my_handler_raises)

        def my_int_handler(_):
            return 42
        assert ujson.ujson_loads(ujson.ujson_dumps(_TestObject('foo'), default_handler=my_int_handler)) == 42

        def my_obj_handler(_):
            return datetime.datetime(2013, 2, 3)
        assert ujson.ujson_loads(ujson.ujson_dumps(datetime.datetime(2013, 2, 3))) == ujson.ujson_loads(ujson.ujson_dumps(_TestObject('foo'), default_handler=my_obj_handler))
        obj_list = [_TestObject('foo'), _TestObject('bar')]
        assert json.loads(json.dumps(obj_list, default=str)) == ujson.ujson_loads(ujson.ujson_dumps(obj_list, default_handler=str))

    def test_encode_object(self):

        class _TestObject:

            def __init__(self, a, b, _c, d) -> None:
                self.a = a
                self.b = b
                self._c = _c
                self.d = d

            def e(self):
                return 5
        test_object = _TestObject(a=1, b=2, _c=3, d=4)
        assert ujson.ujson_loads(ujson.ujson_dumps(test_object)) == {'a': 1, 'b': 2, 'd': 4}

    def test_ujson__name__(self):
        assert ujson.__name__ == 'pandas._libs.json'