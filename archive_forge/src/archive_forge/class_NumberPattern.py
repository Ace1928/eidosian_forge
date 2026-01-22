from __future__ import annotations
import datetime
import decimal
import re
import warnings
from typing import TYPE_CHECKING, Any, cast, overload
from babel.core import Locale, default_locale, get_global
from babel.localedata import LocaleDataDict
class NumberPattern:

    def __init__(self, pattern: str, prefix: tuple[str, str], suffix: tuple[str, str], grouping: tuple[int, int], int_prec: tuple[int, int], frac_prec: tuple[int, int], exp_prec: tuple[int, int] | None, exp_plus: bool | None, number_pattern: str | None=None) -> None:
        self.pattern = pattern
        self.prefix = prefix
        self.suffix = suffix
        self.number_pattern = number_pattern
        self.grouping = grouping
        self.int_prec = int_prec
        self.frac_prec = frac_prec
        self.exp_prec = exp_prec
        self.exp_plus = exp_plus
        self.scale = self.compute_scale()

    def __repr__(self) -> str:
        return f'<{type(self).__name__} {self.pattern!r}>'

    def compute_scale(self) -> Literal[0, 2, 3]:
        """Return the scaling factor to apply to the number before rendering.

        Auto-set to a factor of 2 or 3 if presence of a ``%`` or ``‰`` sign is
        detected in the prefix or suffix of the pattern. Default is to not mess
        with the scale at all and keep it to 0.
        """
        scale = 0
        if '%' in ''.join(self.prefix + self.suffix):
            scale = 2
        elif '‰' in ''.join(self.prefix + self.suffix):
            scale = 3
        return scale

    def scientific_notation_elements(self, value: decimal.Decimal, locale: Locale | str | None, *, numbering_system: Literal['default'] | str='latn') -> tuple[decimal.Decimal, int, str]:
        """ Returns normalized scientific notation components of a value.
        """
        exp = value.adjusted()
        value = value * get_decimal_quantum(exp)
        assert value.adjusted() == 0
        lead_shift = max([1, min(self.int_prec)]) - 1
        exp = exp - lead_shift
        value = value * get_decimal_quantum(-lead_shift)
        exp_sign = ''
        if exp < 0:
            exp_sign = get_minus_sign_symbol(locale, numbering_system=numbering_system)
        elif self.exp_plus:
            exp_sign = get_plus_sign_symbol(locale, numbering_system=numbering_system)
        exp = abs(exp)
        return (value, exp, exp_sign)

    def apply(self, value: float | decimal.Decimal | str, locale: Locale | str | None, currency: str | None=None, currency_digits: bool=True, decimal_quantization: bool=True, force_frac: tuple[int, int] | None=None, group_separator: bool=True, *, numbering_system: Literal['default'] | str='latn'):
        """Renders into a string a number following the defined pattern.

        Forced decimal quantization is active by default so we'll produce a
        number string that is strictly following CLDR pattern definitions.

        :param value: The value to format. If this is not a Decimal object,
                      it will be cast to one.
        :type value: decimal.Decimal|float|int
        :param locale: The locale to use for formatting.
        :type locale: str|babel.core.Locale
        :param currency: Which currency, if any, to format as.
        :type currency: str|None
        :param currency_digits: Whether or not to use the currency's precision.
                                If false, the pattern's precision is used.
        :type currency_digits: bool
        :param decimal_quantization: Whether decimal numbers should be forcibly
                                     quantized to produce a formatted output
                                     strictly matching the CLDR definition for
                                     the locale.
        :type decimal_quantization: bool
        :param force_frac: DEPRECATED - a forced override for `self.frac_prec`
                           for a single formatting invocation.
        :param numbering_system: The numbering system used for formatting number symbols. Defaults to "latn".
                                 The special value "default" will use the default numbering system of the locale.
        :return: Formatted decimal string.
        :rtype: str
        :raise UnsupportedNumberingSystemError: If the numbering system is not supported by the locale.
        """
        if not isinstance(value, decimal.Decimal):
            value = decimal.Decimal(str(value))
        value = value.scaleb(self.scale)
        is_negative = int(value.is_signed())
        value = abs(value).normalize()
        if self.exp_prec:
            value, exp, exp_sign = self.scientific_notation_elements(value, locale, numbering_system=numbering_system)
        if force_frac:
            warnings.warn('The force_frac parameter to NumberPattern.apply() is deprecated.', DeprecationWarning, stacklevel=2)
            frac_prec = force_frac
        elif currency and currency_digits:
            frac_prec = (get_currency_precision(currency),) * 2
        else:
            frac_prec = self.frac_prec
        if not decimal_quantization or (self.exp_prec and frac_prec == (0, 0)):
            frac_prec = (frac_prec[0], max([frac_prec[1], get_decimal_precision(value)]))
        if self.exp_prec:
            number = ''.join([self._quantize_value(value, locale, frac_prec, group_separator, numbering_system=numbering_system), get_exponential_symbol(locale, numbering_system=numbering_system), exp_sign, self._format_int(str(exp), self.exp_prec[0], self.exp_prec[1], locale, numbering_system=numbering_system)])
        elif '@' in self.pattern:
            text = self._format_significant(value, self.int_prec[0], self.int_prec[1])
            a, sep, b = text.partition('.')
            number = self._format_int(a, 0, 1000, locale, numbering_system=numbering_system)
            if sep:
                number += get_decimal_symbol(locale, numbering_system=numbering_system) + b
        else:
            number = self._quantize_value(value, locale, frac_prec, group_separator, numbering_system=numbering_system)
        retval = ''.join([self.prefix[is_negative], number if self.number_pattern != '' else '', self.suffix[is_negative]])
        if '¤' in retval and currency is not None:
            retval = retval.replace('¤¤¤', get_currency_name(currency, value, locale))
            retval = retval.replace('¤¤', currency.upper())
            retval = retval.replace('¤', get_currency_symbol(currency, locale))
        retval = re.sub("'([^']*)'", lambda m: m.group(1) or "'", retval)
        return retval

    def _format_significant(self, value: decimal.Decimal, minimum: int, maximum: int) -> str:
        exp = value.adjusted()
        scale = maximum - 1 - exp
        digits = str(value.scaleb(scale).quantize(decimal.Decimal(1)))
        if scale <= 0:
            result = digits + '0' * -scale
        else:
            intpart = digits[:-scale]
            i = len(intpart)
            j = i + max(minimum - i, 0)
            result = '{intpart}.{pad:0<{fill}}{fracpart}{fracextra}'.format(intpart=intpart or '0', pad='', fill=-min(exp + 1, 0), fracpart=digits[i:j], fracextra=digits[j:].rstrip('0')).rstrip('.')
        return result

    def _format_int(self, value: str, min: int, max: int, locale: Locale | str | None, *, numbering_system: Literal['default'] | str) -> str:
        width = len(value)
        if width < min:
            value = '0' * (min - width) + value
        gsize = self.grouping[0]
        ret = ''
        symbol = get_group_symbol(locale, numbering_system=numbering_system)
        while len(value) > gsize:
            ret = symbol + value[-gsize:] + ret
            value = value[:-gsize]
            gsize = self.grouping[1]
        return value + ret

    def _quantize_value(self, value: decimal.Decimal, locale: Locale | str | None, frac_prec: tuple[int, int], group_separator: bool, *, numbering_system: Literal['default'] | str) -> str:
        if value.is_infinite():
            return get_infinity_symbol(locale, numbering_system=numbering_system)
        quantum = get_decimal_quantum(frac_prec[1])
        rounded = value.quantize(quantum)
        a, sep, b = f'{rounded:f}'.partition('.')
        integer_part = a
        if group_separator:
            integer_part = self._format_int(a, self.int_prec[0], self.int_prec[1], locale, numbering_system=numbering_system)
        number = integer_part + self._format_frac(b or '0', locale=locale, force_frac=frac_prec, numbering_system=numbering_system)
        return number

    def _format_frac(self, value: str, locale: Locale | str | None, force_frac: tuple[int, int] | None=None, *, numbering_system: Literal['default'] | str) -> str:
        min, max = force_frac or self.frac_prec
        if len(value) < min:
            value += '0' * (min - len(value))
        if max == 0 or (min == 0 and int(value) == 0):
            return ''
        while len(value) > min and value[-1] == '0':
            value = value[:-1]
        return get_decimal_symbol(locale, numbering_system=numbering_system) + value