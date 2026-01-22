import math as _math
import numbers as _numbers
import sys
import contextvars
import re
class Decimal(object):
    """Floating point class for decimal arithmetic."""
    __slots__ = ('_exp', '_int', '_sign', '_is_special')

    def __new__(cls, value='0', context=None):
        """Create a decimal point instance.

        >>> Decimal('3.14')              # string input
        Decimal('3.14')
        >>> Decimal((0, (3, 1, 4), -2))  # tuple (sign, digit_tuple, exponent)
        Decimal('3.14')
        >>> Decimal(314)                 # int
        Decimal('314')
        >>> Decimal(Decimal(314))        # another decimal instance
        Decimal('314')
        >>> Decimal('  3.14  \\n')        # leading and trailing whitespace okay
        Decimal('3.14')
        """
        self = object.__new__(cls)
        if isinstance(value, str):
            m = _parser(value.strip().replace('_', ''))
            if m is None:
                if context is None:
                    context = getcontext()
                return context._raise_error(ConversionSyntax, 'Invalid literal for Decimal: %r' % value)
            if m.group('sign') == '-':
                self._sign = 1
            else:
                self._sign = 0
            intpart = m.group('int')
            if intpart is not None:
                fracpart = m.group('frac') or ''
                exp = int(m.group('exp') or '0')
                self._int = str(int(intpart + fracpart))
                self._exp = exp - len(fracpart)
                self._is_special = False
            else:
                diag = m.group('diag')
                if diag is not None:
                    self._int = str(int(diag or '0')).lstrip('0')
                    if m.group('signal'):
                        self._exp = 'N'
                    else:
                        self._exp = 'n'
                else:
                    self._int = '0'
                    self._exp = 'F'
                self._is_special = True
            return self
        if isinstance(value, int):
            if value >= 0:
                self._sign = 0
            else:
                self._sign = 1
            self._exp = 0
            self._int = str(abs(value))
            self._is_special = False
            return self
        if isinstance(value, Decimal):
            self._exp = value._exp
            self._sign = value._sign
            self._int = value._int
            self._is_special = value._is_special
            return self
        if isinstance(value, _WorkRep):
            self._sign = value.sign
            self._int = str(value.int)
            self._exp = int(value.exp)
            self._is_special = False
            return self
        if isinstance(value, (list, tuple)):
            if len(value) != 3:
                raise ValueError('Invalid tuple size in creation of Decimal from list or tuple.  The list or tuple should have exactly three elements.')
            if not (isinstance(value[0], int) and value[0] in (0, 1)):
                raise ValueError('Invalid sign.  The first value in the tuple should be an integer; either 0 for a positive number or 1 for a negative number.')
            self._sign = value[0]
            if value[2] == 'F':
                self._int = '0'
                self._exp = value[2]
                self._is_special = True
            else:
                digits = []
                for digit in value[1]:
                    if isinstance(digit, int) and 0 <= digit <= 9:
                        if digits or digit != 0:
                            digits.append(digit)
                    else:
                        raise ValueError('The second value in the tuple must be composed of integers in the range 0 through 9.')
                if value[2] in ('n', 'N'):
                    self._int = ''.join(map(str, digits))
                    self._exp = value[2]
                    self._is_special = True
                elif isinstance(value[2], int):
                    self._int = ''.join(map(str, digits or [0]))
                    self._exp = value[2]
                    self._is_special = False
                else:
                    raise ValueError("The third value in the tuple must be an integer, or one of the strings 'F', 'n', 'N'.")
            return self
        if isinstance(value, float):
            if context is None:
                context = getcontext()
            context._raise_error(FloatOperation, 'strict semantics for mixing floats and Decimals are enabled')
            value = Decimal.from_float(value)
            self._exp = value._exp
            self._sign = value._sign
            self._int = value._int
            self._is_special = value._is_special
            return self
        raise TypeError('Cannot convert %r to Decimal' % value)

    @classmethod
    def from_float(cls, f):
        """Converts a float to a decimal number, exactly.

        Note that Decimal.from_float(0.1) is not the same as Decimal('0.1').
        Since 0.1 is not exactly representable in binary floating point, the
        value is stored as the nearest representable value which is
        0x1.999999999999ap-4.  The exact equivalent of the value in decimal
        is 0.1000000000000000055511151231257827021181583404541015625.

        >>> Decimal.from_float(0.1)
        Decimal('0.1000000000000000055511151231257827021181583404541015625')
        >>> Decimal.from_float(float('nan'))
        Decimal('NaN')
        >>> Decimal.from_float(float('inf'))
        Decimal('Infinity')
        >>> Decimal.from_float(-float('inf'))
        Decimal('-Infinity')
        >>> Decimal.from_float(-0.0)
        Decimal('-0')

        """
        if isinstance(f, int):
            sign = 0 if f >= 0 else 1
            k = 0
            coeff = str(abs(f))
        elif isinstance(f, float):
            if _math.isinf(f) or _math.isnan(f):
                return cls(repr(f))
            if _math.copysign(1.0, f) == 1.0:
                sign = 0
            else:
                sign = 1
            n, d = abs(f).as_integer_ratio()
            k = d.bit_length() - 1
            coeff = str(n * 5 ** k)
        else:
            raise TypeError('argument must be int or float.')
        result = _dec_from_triple(sign, coeff, -k)
        if cls is Decimal:
            return result
        else:
            return cls(result)

    def _isnan(self):
        """Returns whether the number is not actually one.

        0 if a number
        1 if NaN
        2 if sNaN
        """
        if self._is_special:
            exp = self._exp
            if exp == 'n':
                return 1
            elif exp == 'N':
                return 2
        return 0

    def _isinfinity(self):
        """Returns whether the number is infinite

        0 if finite or not a number
        1 if +INF
        -1 if -INF
        """
        if self._exp == 'F':
            if self._sign:
                return -1
            return 1
        return 0

    def _check_nans(self, other=None, context=None):
        """Returns whether the number is not actually one.

        if self, other are sNaN, signal
        if self, other are NaN return nan
        return 0

        Done before operations.
        """
        self_is_nan = self._isnan()
        if other is None:
            other_is_nan = False
        else:
            other_is_nan = other._isnan()
        if self_is_nan or other_is_nan:
            if context is None:
                context = getcontext()
            if self_is_nan == 2:
                return context._raise_error(InvalidOperation, 'sNaN', self)
            if other_is_nan == 2:
                return context._raise_error(InvalidOperation, 'sNaN', other)
            if self_is_nan:
                return self._fix_nan(context)
            return other._fix_nan(context)
        return 0

    def _compare_check_nans(self, other, context):
        """Version of _check_nans used for the signaling comparisons
        compare_signal, __le__, __lt__, __ge__, __gt__.

        Signal InvalidOperation if either self or other is a (quiet
        or signaling) NaN.  Signaling NaNs take precedence over quiet
        NaNs.

        Return 0 if neither operand is a NaN.

        """
        if context is None:
            context = getcontext()
        if self._is_special or other._is_special:
            if self.is_snan():
                return context._raise_error(InvalidOperation, 'comparison involving sNaN', self)
            elif other.is_snan():
                return context._raise_error(InvalidOperation, 'comparison involving sNaN', other)
            elif self.is_qnan():
                return context._raise_error(InvalidOperation, 'comparison involving NaN', self)
            elif other.is_qnan():
                return context._raise_error(InvalidOperation, 'comparison involving NaN', other)
        return 0

    def __bool__(self):
        """Return True if self is nonzero; otherwise return False.

        NaNs and infinities are considered nonzero.
        """
        return self._is_special or self._int != '0'

    def _cmp(self, other):
        """Compare the two non-NaN decimal instances self and other.

        Returns -1 if self < other, 0 if self == other and 1
        if self > other.  This routine is for internal use only."""
        if self._is_special or other._is_special:
            self_inf = self._isinfinity()
            other_inf = other._isinfinity()
            if self_inf == other_inf:
                return 0
            elif self_inf < other_inf:
                return -1
            else:
                return 1
        if not self:
            if not other:
                return 0
            else:
                return -(-1) ** other._sign
        if not other:
            return (-1) ** self._sign
        if other._sign < self._sign:
            return -1
        if self._sign < other._sign:
            return 1
        self_adjusted = self.adjusted()
        other_adjusted = other.adjusted()
        if self_adjusted == other_adjusted:
            self_padded = self._int + '0' * (self._exp - other._exp)
            other_padded = other._int + '0' * (other._exp - self._exp)
            if self_padded == other_padded:
                return 0
            elif self_padded < other_padded:
                return -(-1) ** self._sign
            else:
                return (-1) ** self._sign
        elif self_adjusted > other_adjusted:
            return (-1) ** self._sign
        else:
            return -(-1) ** self._sign

    def __eq__(self, other, context=None):
        self, other = _convert_for_comparison(self, other, equality_op=True)
        if other is NotImplemented:
            return other
        if self._check_nans(other, context):
            return False
        return self._cmp(other) == 0

    def __lt__(self, other, context=None):
        self, other = _convert_for_comparison(self, other)
        if other is NotImplemented:
            return other
        ans = self._compare_check_nans(other, context)
        if ans:
            return False
        return self._cmp(other) < 0

    def __le__(self, other, context=None):
        self, other = _convert_for_comparison(self, other)
        if other is NotImplemented:
            return other
        ans = self._compare_check_nans(other, context)
        if ans:
            return False
        return self._cmp(other) <= 0

    def __gt__(self, other, context=None):
        self, other = _convert_for_comparison(self, other)
        if other is NotImplemented:
            return other
        ans = self._compare_check_nans(other, context)
        if ans:
            return False
        return self._cmp(other) > 0

    def __ge__(self, other, context=None):
        self, other = _convert_for_comparison(self, other)
        if other is NotImplemented:
            return other
        ans = self._compare_check_nans(other, context)
        if ans:
            return False
        return self._cmp(other) >= 0

    def compare(self, other, context=None):
        """Compare self to other.  Return a decimal value:

        a or b is a NaN ==> Decimal('NaN')
        a < b           ==> Decimal('-1')
        a == b          ==> Decimal('0')
        a > b           ==> Decimal('1')
        """
        other = _convert_other(other, raiseit=True)
        if self._is_special or (other and other._is_special):
            ans = self._check_nans(other, context)
            if ans:
                return ans
        return Decimal(self._cmp(other))

    def __hash__(self):
        """x.__hash__() <==> hash(x)"""
        if self._is_special:
            if self.is_snan():
                raise TypeError('Cannot hash a signaling NaN value.')
            elif self.is_nan():
                return object.__hash__(self)
            elif self._sign:
                return -_PyHASH_INF
            else:
                return _PyHASH_INF
        if self._exp >= 0:
            exp_hash = pow(10, self._exp, _PyHASH_MODULUS)
        else:
            exp_hash = pow(_PyHASH_10INV, -self._exp, _PyHASH_MODULUS)
        hash_ = int(self._int) * exp_hash % _PyHASH_MODULUS
        ans = hash_ if self >= 0 else -hash_
        return -2 if ans == -1 else ans

    def as_tuple(self):
        """Represents the number as a triple tuple.

        To show the internals exactly as they are.
        """
        return DecimalTuple(self._sign, tuple(map(int, self._int)), self._exp)

    def as_integer_ratio(self):
        """Express a finite Decimal instance in the form n / d.

        Returns a pair (n, d) of integers.  When called on an infinity
        or NaN, raises OverflowError or ValueError respectively.

        >>> Decimal('3.14').as_integer_ratio()
        (157, 50)
        >>> Decimal('-123e5').as_integer_ratio()
        (-12300000, 1)
        >>> Decimal('0.00').as_integer_ratio()
        (0, 1)

        """
        if self._is_special:
            if self.is_nan():
                raise ValueError('cannot convert NaN to integer ratio')
            else:
                raise OverflowError('cannot convert Infinity to integer ratio')
        if not self:
            return (0, 1)
        n = int(self._int)
        if self._exp >= 0:
            n, d = (n * 10 ** self._exp, 1)
        else:
            d5 = -self._exp
            while d5 > 0 and n % 5 == 0:
                n //= 5
                d5 -= 1
            d2 = -self._exp
            shift2 = min((n & -n).bit_length() - 1, d2)
            if shift2:
                n >>= shift2
                d2 -= shift2
            d = 5 ** d5 << d2
        if self._sign:
            n = -n
        return (n, d)

    def __repr__(self):
        """Represents the number as an instance of Decimal."""
        return "Decimal('%s')" % str(self)

    def __str__(self, eng=False, context=None):
        """Return string representation of the number in scientific notation.

        Captures all of the information in the underlying representation.
        """
        sign = ['', '-'][self._sign]
        if self._is_special:
            if self._exp == 'F':
                return sign + 'Infinity'
            elif self._exp == 'n':
                return sign + 'NaN' + self._int
            else:
                return sign + 'sNaN' + self._int
        leftdigits = self._exp + len(self._int)
        if self._exp <= 0 and leftdigits > -6:
            dotplace = leftdigits
        elif not eng:
            dotplace = 1
        elif self._int == '0':
            dotplace = (leftdigits + 1) % 3 - 1
        else:
            dotplace = (leftdigits - 1) % 3 + 1
        if dotplace <= 0:
            intpart = '0'
            fracpart = '.' + '0' * -dotplace + self._int
        elif dotplace >= len(self._int):
            intpart = self._int + '0' * (dotplace - len(self._int))
            fracpart = ''
        else:
            intpart = self._int[:dotplace]
            fracpart = '.' + self._int[dotplace:]
        if leftdigits == dotplace:
            exp = ''
        else:
            if context is None:
                context = getcontext()
            exp = ['e', 'E'][context.capitals] + '%+d' % (leftdigits - dotplace)
        return sign + intpart + fracpart + exp

    def to_eng_string(self, context=None):
        """Convert to a string, using engineering notation if an exponent is needed.

        Engineering notation has an exponent which is a multiple of 3.  This
        can leave up to 3 digits to the left of the decimal place and may
        require the addition of either one or two trailing zeros.
        """
        return self.__str__(eng=True, context=context)

    def __neg__(self, context=None):
        """Returns a copy with the sign switched.

        Rounds, if it has reason.
        """
        if self._is_special:
            ans = self._check_nans(context=context)
            if ans:
                return ans
        if context is None:
            context = getcontext()
        if not self and context.rounding != ROUND_FLOOR:
            ans = self.copy_abs()
        else:
            ans = self.copy_negate()
        return ans._fix(context)

    def __pos__(self, context=None):
        """Returns a copy, unless it is a sNaN.

        Rounds the number (if more than precision digits)
        """
        if self._is_special:
            ans = self._check_nans(context=context)
            if ans:
                return ans
        if context is None:
            context = getcontext()
        if not self and context.rounding != ROUND_FLOOR:
            ans = self.copy_abs()
        else:
            ans = Decimal(self)
        return ans._fix(context)

    def __abs__(self, round=True, context=None):
        """Returns the absolute value of self.

        If the keyword argument 'round' is false, do not round.  The
        expression self.__abs__(round=False) is equivalent to
        self.copy_abs().
        """
        if not round:
            return self.copy_abs()
        if self._is_special:
            ans = self._check_nans(context=context)
            if ans:
                return ans
        if self._sign:
            ans = self.__neg__(context=context)
        else:
            ans = self.__pos__(context=context)
        return ans

    def __add__(self, other, context=None):
        """Returns self + other.

        -INF + INF (or the reverse) cause InvalidOperation errors.
        """
        other = _convert_other(other)
        if other is NotImplemented:
            return other
        if context is None:
            context = getcontext()
        if self._is_special or other._is_special:
            ans = self._check_nans(other, context)
            if ans:
                return ans
            if self._isinfinity():
                if self._sign != other._sign and other._isinfinity():
                    return context._raise_error(InvalidOperation, '-INF + INF')
                return Decimal(self)
            if other._isinfinity():
                return Decimal(other)
        exp = min(self._exp, other._exp)
        negativezero = 0
        if context.rounding == ROUND_FLOOR and self._sign != other._sign:
            negativezero = 1
        if not self and (not other):
            sign = min(self._sign, other._sign)
            if negativezero:
                sign = 1
            ans = _dec_from_triple(sign, '0', exp)
            ans = ans._fix(context)
            return ans
        if not self:
            exp = max(exp, other._exp - context.prec - 1)
            ans = other._rescale(exp, context.rounding)
            ans = ans._fix(context)
            return ans
        if not other:
            exp = max(exp, self._exp - context.prec - 1)
            ans = self._rescale(exp, context.rounding)
            ans = ans._fix(context)
            return ans
        op1 = _WorkRep(self)
        op2 = _WorkRep(other)
        op1, op2 = _normalize(op1, op2, context.prec)
        result = _WorkRep()
        if op1.sign != op2.sign:
            if op1.int == op2.int:
                ans = _dec_from_triple(negativezero, '0', exp)
                ans = ans._fix(context)
                return ans
            if op1.int < op2.int:
                op1, op2 = (op2, op1)
            if op1.sign == 1:
                result.sign = 1
                op1.sign, op2.sign = (op2.sign, op1.sign)
            else:
                result.sign = 0
        elif op1.sign == 1:
            result.sign = 1
            op1.sign, op2.sign = (0, 0)
        else:
            result.sign = 0
        if op2.sign == 0:
            result.int = op1.int + op2.int
        else:
            result.int = op1.int - op2.int
        result.exp = op1.exp
        ans = Decimal(result)
        ans = ans._fix(context)
        return ans
    __radd__ = __add__

    def __sub__(self, other, context=None):
        """Return self - other"""
        other = _convert_other(other)
        if other is NotImplemented:
            return other
        if self._is_special or other._is_special:
            ans = self._check_nans(other, context=context)
            if ans:
                return ans
        return self.__add__(other.copy_negate(), context=context)

    def __rsub__(self, other, context=None):
        """Return other - self"""
        other = _convert_other(other)
        if other is NotImplemented:
            return other
        return other.__sub__(self, context=context)

    def __mul__(self, other, context=None):
        """Return self * other.

        (+-) INF * 0 (or its reverse) raise InvalidOperation.
        """
        other = _convert_other(other)
        if other is NotImplemented:
            return other
        if context is None:
            context = getcontext()
        resultsign = self._sign ^ other._sign
        if self._is_special or other._is_special:
            ans = self._check_nans(other, context)
            if ans:
                return ans
            if self._isinfinity():
                if not other:
                    return context._raise_error(InvalidOperation, '(+-)INF * 0')
                return _SignedInfinity[resultsign]
            if other._isinfinity():
                if not self:
                    return context._raise_error(InvalidOperation, '0 * (+-)INF')
                return _SignedInfinity[resultsign]
        resultexp = self._exp + other._exp
        if not self or not other:
            ans = _dec_from_triple(resultsign, '0', resultexp)
            ans = ans._fix(context)
            return ans
        if self._int == '1':
            ans = _dec_from_triple(resultsign, other._int, resultexp)
            ans = ans._fix(context)
            return ans
        if other._int == '1':
            ans = _dec_from_triple(resultsign, self._int, resultexp)
            ans = ans._fix(context)
            return ans
        op1 = _WorkRep(self)
        op2 = _WorkRep(other)
        ans = _dec_from_triple(resultsign, str(op1.int * op2.int), resultexp)
        ans = ans._fix(context)
        return ans
    __rmul__ = __mul__

    def __truediv__(self, other, context=None):
        """Return self / other."""
        other = _convert_other(other)
        if other is NotImplemented:
            return NotImplemented
        if context is None:
            context = getcontext()
        sign = self._sign ^ other._sign
        if self._is_special or other._is_special:
            ans = self._check_nans(other, context)
            if ans:
                return ans
            if self._isinfinity() and other._isinfinity():
                return context._raise_error(InvalidOperation, '(+-)INF/(+-)INF')
            if self._isinfinity():
                return _SignedInfinity[sign]
            if other._isinfinity():
                context._raise_error(Clamped, 'Division by infinity')
                return _dec_from_triple(sign, '0', context.Etiny())
        if not other:
            if not self:
                return context._raise_error(DivisionUndefined, '0 / 0')
            return context._raise_error(DivisionByZero, 'x / 0', sign)
        if not self:
            exp = self._exp - other._exp
            coeff = 0
        else:
            shift = len(other._int) - len(self._int) + context.prec + 1
            exp = self._exp - other._exp - shift
            op1 = _WorkRep(self)
            op2 = _WorkRep(other)
            if shift >= 0:
                coeff, remainder = divmod(op1.int * 10 ** shift, op2.int)
            else:
                coeff, remainder = divmod(op1.int, op2.int * 10 ** (-shift))
            if remainder:
                if coeff % 5 == 0:
                    coeff += 1
            else:
                ideal_exp = self._exp - other._exp
                while exp < ideal_exp and coeff % 10 == 0:
                    coeff //= 10
                    exp += 1
        ans = _dec_from_triple(sign, str(coeff), exp)
        return ans._fix(context)

    def _divide(self, other, context):
        """Return (self // other, self % other), to context.prec precision.

        Assumes that neither self nor other is a NaN, that self is not
        infinite and that other is nonzero.
        """
        sign = self._sign ^ other._sign
        if other._isinfinity():
            ideal_exp = self._exp
        else:
            ideal_exp = min(self._exp, other._exp)
        expdiff = self.adjusted() - other.adjusted()
        if not self or other._isinfinity() or expdiff <= -2:
            return (_dec_from_triple(sign, '0', 0), self._rescale(ideal_exp, context.rounding))
        if expdiff <= context.prec:
            op1 = _WorkRep(self)
            op2 = _WorkRep(other)
            if op1.exp >= op2.exp:
                op1.int *= 10 ** (op1.exp - op2.exp)
            else:
                op2.int *= 10 ** (op2.exp - op1.exp)
            q, r = divmod(op1.int, op2.int)
            if q < 10 ** context.prec:
                return (_dec_from_triple(sign, str(q), 0), _dec_from_triple(self._sign, str(r), ideal_exp))
        ans = context._raise_error(DivisionImpossible, 'quotient too large in //, % or divmod')
        return (ans, ans)

    def __rtruediv__(self, other, context=None):
        """Swaps self/other and returns __truediv__."""
        other = _convert_other(other)
        if other is NotImplemented:
            return other
        return other.__truediv__(self, context=context)

    def __divmod__(self, other, context=None):
        """
        Return (self // other, self % other)
        """
        other = _convert_other(other)
        if other is NotImplemented:
            return other
        if context is None:
            context = getcontext()
        ans = self._check_nans(other, context)
        if ans:
            return (ans, ans)
        sign = self._sign ^ other._sign
        if self._isinfinity():
            if other._isinfinity():
                ans = context._raise_error(InvalidOperation, 'divmod(INF, INF)')
                return (ans, ans)
            else:
                return (_SignedInfinity[sign], context._raise_error(InvalidOperation, 'INF % x'))
        if not other:
            if not self:
                ans = context._raise_error(DivisionUndefined, 'divmod(0, 0)')
                return (ans, ans)
            else:
                return (context._raise_error(DivisionByZero, 'x // 0', sign), context._raise_error(InvalidOperation, 'x % 0'))
        quotient, remainder = self._divide(other, context)
        remainder = remainder._fix(context)
        return (quotient, remainder)

    def __rdivmod__(self, other, context=None):
        """Swaps self/other and returns __divmod__."""
        other = _convert_other(other)
        if other is NotImplemented:
            return other
        return other.__divmod__(self, context=context)

    def __mod__(self, other, context=None):
        """
        self % other
        """
        other = _convert_other(other)
        if other is NotImplemented:
            return other
        if context is None:
            context = getcontext()
        ans = self._check_nans(other, context)
        if ans:
            return ans
        if self._isinfinity():
            return context._raise_error(InvalidOperation, 'INF % x')
        elif not other:
            if self:
                return context._raise_error(InvalidOperation, 'x % 0')
            else:
                return context._raise_error(DivisionUndefined, '0 % 0')
        remainder = self._divide(other, context)[1]
        remainder = remainder._fix(context)
        return remainder

    def __rmod__(self, other, context=None):
        """Swaps self/other and returns __mod__."""
        other = _convert_other(other)
        if other is NotImplemented:
            return other
        return other.__mod__(self, context=context)

    def remainder_near(self, other, context=None):
        """
        Remainder nearest to 0-  abs(remainder-near) <= other/2
        """
        if context is None:
            context = getcontext()
        other = _convert_other(other, raiseit=True)
        ans = self._check_nans(other, context)
        if ans:
            return ans
        if self._isinfinity():
            return context._raise_error(InvalidOperation, 'remainder_near(infinity, x)')
        if not other:
            if self:
                return context._raise_error(InvalidOperation, 'remainder_near(x, 0)')
            else:
                return context._raise_error(DivisionUndefined, 'remainder_near(0, 0)')
        if other._isinfinity():
            ans = Decimal(self)
            return ans._fix(context)
        ideal_exponent = min(self._exp, other._exp)
        if not self:
            ans = _dec_from_triple(self._sign, '0', ideal_exponent)
            return ans._fix(context)
        expdiff = self.adjusted() - other.adjusted()
        if expdiff >= context.prec + 1:
            return context._raise_error(DivisionImpossible)
        if expdiff <= -2:
            ans = self._rescale(ideal_exponent, context.rounding)
            return ans._fix(context)
        op1 = _WorkRep(self)
        op2 = _WorkRep(other)
        if op1.exp >= op2.exp:
            op1.int *= 10 ** (op1.exp - op2.exp)
        else:
            op2.int *= 10 ** (op2.exp - op1.exp)
        q, r = divmod(op1.int, op2.int)
        if 2 * r + (q & 1) > op2.int:
            r -= op2.int
            q += 1
        if q >= 10 ** context.prec:
            return context._raise_error(DivisionImpossible)
        sign = self._sign
        if r < 0:
            sign = 1 - sign
            r = -r
        ans = _dec_from_triple(sign, str(r), ideal_exponent)
        return ans._fix(context)

    def __floordiv__(self, other, context=None):
        """self // other"""
        other = _convert_other(other)
        if other is NotImplemented:
            return other
        if context is None:
            context = getcontext()
        ans = self._check_nans(other, context)
        if ans:
            return ans
        if self._isinfinity():
            if other._isinfinity():
                return context._raise_error(InvalidOperation, 'INF // INF')
            else:
                return _SignedInfinity[self._sign ^ other._sign]
        if not other:
            if self:
                return context._raise_error(DivisionByZero, 'x // 0', self._sign ^ other._sign)
            else:
                return context._raise_error(DivisionUndefined, '0 // 0')
        return self._divide(other, context)[0]

    def __rfloordiv__(self, other, context=None):
        """Swaps self/other and returns __floordiv__."""
        other = _convert_other(other)
        if other is NotImplemented:
            return other
        return other.__floordiv__(self, context=context)

    def __float__(self):
        """Float representation."""
        if self._isnan():
            if self.is_snan():
                raise ValueError('Cannot convert signaling NaN to float')
            s = '-nan' if self._sign else 'nan'
        else:
            s = str(self)
        return float(s)

    def __int__(self):
        """Converts self to an int, truncating if necessary."""
        if self._is_special:
            if self._isnan():
                raise ValueError('Cannot convert NaN to integer')
            elif self._isinfinity():
                raise OverflowError('Cannot convert infinity to integer')
        s = (-1) ** self._sign
        if self._exp >= 0:
            return s * int(self._int) * 10 ** self._exp
        else:
            return s * int(self._int[:self._exp] or '0')
    __trunc__ = __int__

    @property
    def real(self):
        return self

    @property
    def imag(self):
        return Decimal(0)

    def conjugate(self):
        return self

    def __complex__(self):
        return complex(float(self))

    def _fix_nan(self, context):
        """Decapitate the payload of a NaN to fit the context"""
        payload = self._int
        max_payload_len = context.prec - context.clamp
        if len(payload) > max_payload_len:
            payload = payload[len(payload) - max_payload_len:].lstrip('0')
            return _dec_from_triple(self._sign, payload, self._exp, True)
        return Decimal(self)

    def _fix(self, context):
        """Round if it is necessary to keep self within prec precision.

        Rounds and fixes the exponent.  Does not raise on a sNaN.

        Arguments:
        self - Decimal instance
        context - context used.
        """
        if self._is_special:
            if self._isnan():
                return self._fix_nan(context)
            else:
                return Decimal(self)
        Etiny = context.Etiny()
        Etop = context.Etop()
        if not self:
            exp_max = [context.Emax, Etop][context.clamp]
            new_exp = min(max(self._exp, Etiny), exp_max)
            if new_exp != self._exp:
                context._raise_error(Clamped)
                return _dec_from_triple(self._sign, '0', new_exp)
            else:
                return Decimal(self)
        exp_min = len(self._int) + self._exp - context.prec
        if exp_min > Etop:
            ans = context._raise_error(Overflow, 'above Emax', self._sign)
            context._raise_error(Inexact)
            context._raise_error(Rounded)
            return ans
        self_is_subnormal = exp_min < Etiny
        if self_is_subnormal:
            exp_min = Etiny
        if self._exp < exp_min:
            digits = len(self._int) + self._exp - exp_min
            if digits < 0:
                self = _dec_from_triple(self._sign, '1', exp_min - 1)
                digits = 0
            rounding_method = self._pick_rounding_function[context.rounding]
            changed = rounding_method(self, digits)
            coeff = self._int[:digits] or '0'
            if changed > 0:
                coeff = str(int(coeff) + 1)
                if len(coeff) > context.prec:
                    coeff = coeff[:-1]
                    exp_min += 1
            if exp_min > Etop:
                ans = context._raise_error(Overflow, 'above Emax', self._sign)
            else:
                ans = _dec_from_triple(self._sign, coeff, exp_min)
            if changed and self_is_subnormal:
                context._raise_error(Underflow)
            if self_is_subnormal:
                context._raise_error(Subnormal)
            if changed:
                context._raise_error(Inexact)
            context._raise_error(Rounded)
            if not ans:
                context._raise_error(Clamped)
            return ans
        if self_is_subnormal:
            context._raise_error(Subnormal)
        if context.clamp == 1 and self._exp > Etop:
            context._raise_error(Clamped)
            self_padded = self._int + '0' * (self._exp - Etop)
            return _dec_from_triple(self._sign, self_padded, Etop)
        return Decimal(self)

    def _round_down(self, prec):
        """Also known as round-towards-0, truncate."""
        if _all_zeros(self._int, prec):
            return 0
        else:
            return -1

    def _round_up(self, prec):
        """Rounds away from 0."""
        return -self._round_down(prec)

    def _round_half_up(self, prec):
        """Rounds 5 up (away from 0)"""
        if self._int[prec] in '56789':
            return 1
        elif _all_zeros(self._int, prec):
            return 0
        else:
            return -1

    def _round_half_down(self, prec):
        """Round 5 down"""
        if _exact_half(self._int, prec):
            return -1
        else:
            return self._round_half_up(prec)

    def _round_half_even(self, prec):
        """Round 5 to even, rest to nearest."""
        if _exact_half(self._int, prec) and (prec == 0 or self._int[prec - 1] in '02468'):
            return -1
        else:
            return self._round_half_up(prec)

    def _round_ceiling(self, prec):
        """Rounds up (not away from 0 if negative.)"""
        if self._sign:
            return self._round_down(prec)
        else:
            return -self._round_down(prec)

    def _round_floor(self, prec):
        """Rounds down (not towards 0 if negative)"""
        if not self._sign:
            return self._round_down(prec)
        else:
            return -self._round_down(prec)

    def _round_05up(self, prec):
        """Round down unless digit prec-1 is 0 or 5."""
        if prec and self._int[prec - 1] not in '05':
            return self._round_down(prec)
        else:
            return -self._round_down(prec)
    _pick_rounding_function = dict(ROUND_DOWN=_round_down, ROUND_UP=_round_up, ROUND_HALF_UP=_round_half_up, ROUND_HALF_DOWN=_round_half_down, ROUND_HALF_EVEN=_round_half_even, ROUND_CEILING=_round_ceiling, ROUND_FLOOR=_round_floor, ROUND_05UP=_round_05up)

    def __round__(self, n=None):
        """Round self to the nearest integer, or to a given precision.

        If only one argument is supplied, round a finite Decimal
        instance self to the nearest integer.  If self is infinite or
        a NaN then a Python exception is raised.  If self is finite
        and lies exactly halfway between two integers then it is
        rounded to the integer with even last digit.

        >>> round(Decimal('123.456'))
        123
        >>> round(Decimal('-456.789'))
        -457
        >>> round(Decimal('-3.0'))
        -3
        >>> round(Decimal('2.5'))
        2
        >>> round(Decimal('3.5'))
        4
        >>> round(Decimal('Inf'))
        Traceback (most recent call last):
          ...
        OverflowError: cannot round an infinity
        >>> round(Decimal('NaN'))
        Traceback (most recent call last):
          ...
        ValueError: cannot round a NaN

        If a second argument n is supplied, self is rounded to n
        decimal places using the rounding mode for the current
        context.

        For an integer n, round(self, -n) is exactly equivalent to
        self.quantize(Decimal('1En')).

        >>> round(Decimal('123.456'), 0)
        Decimal('123')
        >>> round(Decimal('123.456'), 2)
        Decimal('123.46')
        >>> round(Decimal('123.456'), -2)
        Decimal('1E+2')
        >>> round(Decimal('-Infinity'), 37)
        Decimal('NaN')
        >>> round(Decimal('sNaN123'), 0)
        Decimal('NaN123')

        """
        if n is not None:
            if not isinstance(n, int):
                raise TypeError('Second argument to round should be integral')
            exp = _dec_from_triple(0, '1', -n)
            return self.quantize(exp)
        if self._is_special:
            if self.is_nan():
                raise ValueError('cannot round a NaN')
            else:
                raise OverflowError('cannot round an infinity')
        return int(self._rescale(0, ROUND_HALF_EVEN))

    def __floor__(self):
        """Return the floor of self, as an integer.

        For a finite Decimal instance self, return the greatest
        integer n such that n <= self.  If self is infinite or a NaN
        then a Python exception is raised.

        """
        if self._is_special:
            if self.is_nan():
                raise ValueError('cannot round a NaN')
            else:
                raise OverflowError('cannot round an infinity')
        return int(self._rescale(0, ROUND_FLOOR))

    def __ceil__(self):
        """Return the ceiling of self, as an integer.

        For a finite Decimal instance self, return the least integer n
        such that n >= self.  If self is infinite or a NaN then a
        Python exception is raised.

        """
        if self._is_special:
            if self.is_nan():
                raise ValueError('cannot round a NaN')
            else:
                raise OverflowError('cannot round an infinity')
        return int(self._rescale(0, ROUND_CEILING))

    def fma(self, other, third, context=None):
        """Fused multiply-add.

        Returns self*other+third with no rounding of the intermediate
        product self*other.

        self and other are multiplied together, with no rounding of
        the result.  The third operand is then added to the result,
        and a single final rounding is performed.
        """
        other = _convert_other(other, raiseit=True)
        third = _convert_other(third, raiseit=True)
        if self._is_special or other._is_special:
            if context is None:
                context = getcontext()
            if self._exp == 'N':
                return context._raise_error(InvalidOperation, 'sNaN', self)
            if other._exp == 'N':
                return context._raise_error(InvalidOperation, 'sNaN', other)
            if self._exp == 'n':
                product = self
            elif other._exp == 'n':
                product = other
            elif self._exp == 'F':
                if not other:
                    return context._raise_error(InvalidOperation, 'INF * 0 in fma')
                product = _SignedInfinity[self._sign ^ other._sign]
            elif other._exp == 'F':
                if not self:
                    return context._raise_error(InvalidOperation, '0 * INF in fma')
                product = _SignedInfinity[self._sign ^ other._sign]
        else:
            product = _dec_from_triple(self._sign ^ other._sign, str(int(self._int) * int(other._int)), self._exp + other._exp)
        return product.__add__(third, context)

    def _power_modulo(self, other, modulo, context=None):
        """Three argument version of __pow__"""
        other = _convert_other(other)
        if other is NotImplemented:
            return other
        modulo = _convert_other(modulo)
        if modulo is NotImplemented:
            return modulo
        if context is None:
            context = getcontext()
        self_is_nan = self._isnan()
        other_is_nan = other._isnan()
        modulo_is_nan = modulo._isnan()
        if self_is_nan or other_is_nan or modulo_is_nan:
            if self_is_nan == 2:
                return context._raise_error(InvalidOperation, 'sNaN', self)
            if other_is_nan == 2:
                return context._raise_error(InvalidOperation, 'sNaN', other)
            if modulo_is_nan == 2:
                return context._raise_error(InvalidOperation, 'sNaN', modulo)
            if self_is_nan:
                return self._fix_nan(context)
            if other_is_nan:
                return other._fix_nan(context)
            return modulo._fix_nan(context)
        if not (self._isinteger() and other._isinteger() and modulo._isinteger()):
            return context._raise_error(InvalidOperation, 'pow() 3rd argument not allowed unless all arguments are integers')
        if other < 0:
            return context._raise_error(InvalidOperation, 'pow() 2nd argument cannot be negative when 3rd argument specified')
        if not modulo:
            return context._raise_error(InvalidOperation, 'pow() 3rd argument cannot be 0')
        if modulo.adjusted() >= context.prec:
            return context._raise_error(InvalidOperation, 'insufficient precision: pow() 3rd argument must not have more than precision digits')
        if not other and (not self):
            return context._raise_error(InvalidOperation, 'at least one of pow() 1st argument and 2nd argument must be nonzero; 0**0 is not defined')
        if other._iseven():
            sign = 0
        else:
            sign = self._sign
        modulo = abs(int(modulo))
        base = _WorkRep(self.to_integral_value())
        exponent = _WorkRep(other.to_integral_value())
        base = base.int % modulo * pow(10, base.exp, modulo) % modulo
        for i in range(exponent.exp):
            base = pow(base, 10, modulo)
        base = pow(base, exponent.int, modulo)
        return _dec_from_triple(sign, str(base), 0)

    def _power_exact(self, other, p):
        """Attempt to compute self**other exactly.

        Given Decimals self and other and an integer p, attempt to
        compute an exact result for the power self**other, with p
        digits of precision.  Return None if self**other is not
        exactly representable in p digits.

        Assumes that elimination of special cases has already been
        performed: self and other must both be nonspecial; self must
        be positive and not numerically equal to 1; other must be
        nonzero.  For efficiency, other._exp should not be too large,
        so that 10**abs(other._exp) is a feasible calculation."""
        x = _WorkRep(self)
        xc, xe = (x.int, x.exp)
        while xc % 10 == 0:
            xc //= 10
            xe += 1
        y = _WorkRep(other)
        yc, ye = (y.int, y.exp)
        while yc % 10 == 0:
            yc //= 10
            ye += 1
        if xc == 1:
            xe *= yc
            while xe % 10 == 0:
                xe //= 10
                ye += 1
            if ye < 0:
                return None
            exponent = xe * 10 ** ye
            if y.sign == 1:
                exponent = -exponent
            if other._isinteger() and other._sign == 0:
                ideal_exponent = self._exp * int(other)
                zeros = min(exponent - ideal_exponent, p - 1)
            else:
                zeros = 0
            return _dec_from_triple(0, '1' + '0' * zeros, exponent - zeros)
        if y.sign == 1:
            last_digit = xc % 10
            if last_digit in (2, 4, 6, 8):
                if xc & -xc != xc:
                    return None
                e = _nbits(xc) - 1
                emax = p * 93 // 65
                if ye >= len(str(emax)):
                    return None
                e = _decimal_lshift_exact(e * yc, ye)
                xe = _decimal_lshift_exact(xe * yc, ye)
                if e is None or xe is None:
                    return None
                if e > emax:
                    return None
                xc = 5 ** e
            elif last_digit == 5:
                e = _nbits(xc) * 28 // 65
                xc, remainder = divmod(5 ** e, xc)
                if remainder:
                    return None
                while xc % 5 == 0:
                    xc //= 5
                    e -= 1
                emax = p * 10 // 3
                if ye >= len(str(emax)):
                    return None
                e = _decimal_lshift_exact(e * yc, ye)
                xe = _decimal_lshift_exact(xe * yc, ye)
                if e is None or xe is None:
                    return None
                if e > emax:
                    return None
                xc = 2 ** e
            else:
                return None
            if xc >= 10 ** p:
                return None
            xe = -e - xe
            return _dec_from_triple(0, str(xc), xe)
        if ye >= 0:
            m, n = (yc * 10 ** ye, 1)
        else:
            if xe != 0 and len(str(abs(yc * xe))) <= -ye:
                return None
            xc_bits = _nbits(xc)
            if len(str(abs(yc) * xc_bits)) <= -ye:
                return None
            m, n = (yc, 10 ** (-ye))
            while m % 2 == n % 2 == 0:
                m //= 2
                n //= 2
            while m % 5 == n % 5 == 0:
                m //= 5
                n //= 5
        if n > 1:
            if xc_bits <= n:
                return None
            xe, rem = divmod(xe, n)
            if rem != 0:
                return None
            a = 1 << -(-_nbits(xc) // n)
            while True:
                q, r = divmod(xc, a ** (n - 1))
                if a <= q:
                    break
                else:
                    a = (a * (n - 1) + q) // n
            if not (a == q and r == 0):
                return None
            xc = a
        if xc > 1 and m > p * 100 // _log10_lb(xc):
            return None
        xc = xc ** m
        xe *= m
        if xc > 10 ** p:
            return None
        str_xc = str(xc)
        if other._isinteger() and other._sign == 0:
            ideal_exponent = self._exp * int(other)
            zeros = min(xe - ideal_exponent, p - len(str_xc))
        else:
            zeros = 0
        return _dec_from_triple(0, str_xc + '0' * zeros, xe - zeros)

    def __pow__(self, other, modulo=None, context=None):
        """Return self ** other [ % modulo].

        With two arguments, compute self**other.

        With three arguments, compute (self**other) % modulo.  For the
        three argument form, the following restrictions on the
        arguments hold:

         - all three arguments must be integral
         - other must be nonnegative
         - either self or other (or both) must be nonzero
         - modulo must be nonzero and must have at most p digits,
           where p is the context precision.

        If any of these restrictions is violated the InvalidOperation
        flag is raised.

        The result of pow(self, other, modulo) is identical to the
        result that would be obtained by computing (self**other) %
        modulo with unbounded precision, but is computed more
        efficiently.  It is always exact.
        """
        if modulo is not None:
            return self._power_modulo(other, modulo, context)
        other = _convert_other(other)
        if other is NotImplemented:
            return other
        if context is None:
            context = getcontext()
        ans = self._check_nans(other, context)
        if ans:
            return ans
        if not other:
            if not self:
                return context._raise_error(InvalidOperation, '0 ** 0')
            else:
                return _One
        result_sign = 0
        if self._sign == 1:
            if other._isinteger():
                if not other._iseven():
                    result_sign = 1
            elif self:
                return context._raise_error(InvalidOperation, 'x ** y with x negative and y not an integer')
            self = self.copy_negate()
        if not self:
            if other._sign == 0:
                return _dec_from_triple(result_sign, '0', 0)
            else:
                return _SignedInfinity[result_sign]
        if self._isinfinity():
            if other._sign == 0:
                return _SignedInfinity[result_sign]
            else:
                return _dec_from_triple(result_sign, '0', 0)
        if self == _One:
            if other._isinteger():
                if other._sign == 1:
                    multiplier = 0
                elif other > context.prec:
                    multiplier = context.prec
                else:
                    multiplier = int(other)
                exp = self._exp * multiplier
                if exp < 1 - context.prec:
                    exp = 1 - context.prec
                    context._raise_error(Rounded)
            else:
                context._raise_error(Inexact)
                context._raise_error(Rounded)
                exp = 1 - context.prec
            return _dec_from_triple(result_sign, '1' + '0' * -exp, exp)
        self_adj = self.adjusted()
        if other._isinfinity():
            if (other._sign == 0) == (self_adj < 0):
                return _dec_from_triple(result_sign, '0', 0)
            else:
                return _SignedInfinity[result_sign]
        ans = None
        exact = False
        bound = self._log10_exp_bound() + other.adjusted()
        if (self_adj >= 0) == (other._sign == 0):
            if bound >= len(str(context.Emax)):
                ans = _dec_from_triple(result_sign, '1', context.Emax + 1)
        else:
            Etiny = context.Etiny()
            if bound >= len(str(-Etiny)):
                ans = _dec_from_triple(result_sign, '1', Etiny - 1)
        if ans is None:
            ans = self._power_exact(other, context.prec + 1)
            if ans is not None:
                if result_sign == 1:
                    ans = _dec_from_triple(1, ans._int, ans._exp)
                exact = True
        if ans is None:
            p = context.prec
            x = _WorkRep(self)
            xc, xe = (x.int, x.exp)
            y = _WorkRep(other)
            yc, ye = (y.int, y.exp)
            if y.sign == 1:
                yc = -yc
            extra = 3
            while True:
                coeff, exp = _dpower(xc, xe, yc, ye, p + extra)
                if coeff % (5 * 10 ** (len(str(coeff)) - p - 1)):
                    break
                extra += 3
            ans = _dec_from_triple(result_sign, str(coeff), exp)
        if exact and (not other._isinteger()):
            if len(ans._int) <= context.prec:
                expdiff = context.prec + 1 - len(ans._int)
                ans = _dec_from_triple(ans._sign, ans._int + '0' * expdiff, ans._exp - expdiff)
            newcontext = context.copy()
            newcontext.clear_flags()
            for exception in _signals:
                newcontext.traps[exception] = 0
            ans = ans._fix(newcontext)
            newcontext._raise_error(Inexact)
            if newcontext.flags[Subnormal]:
                newcontext._raise_error(Underflow)
            if newcontext.flags[Overflow]:
                context._raise_error(Overflow, 'above Emax', ans._sign)
            for exception in (Underflow, Subnormal, Inexact, Rounded, Clamped):
                if newcontext.flags[exception]:
                    context._raise_error(exception)
        else:
            ans = ans._fix(context)
        return ans

    def __rpow__(self, other, context=None):
        """Swaps self/other and returns __pow__."""
        other = _convert_other(other)
        if other is NotImplemented:
            return other
        return other.__pow__(self, context=context)

    def normalize(self, context=None):
        """Normalize- strip trailing 0s, change anything equal to 0 to 0e0"""
        if context is None:
            context = getcontext()
        if self._is_special:
            ans = self._check_nans(context=context)
            if ans:
                return ans
        dup = self._fix(context)
        if dup._isinfinity():
            return dup
        if not dup:
            return _dec_from_triple(dup._sign, '0', 0)
        exp_max = [context.Emax, context.Etop()][context.clamp]
        end = len(dup._int)
        exp = dup._exp
        while dup._int[end - 1] == '0' and exp < exp_max:
            exp += 1
            end -= 1
        return _dec_from_triple(dup._sign, dup._int[:end], exp)

    def quantize(self, exp, rounding=None, context=None):
        """Quantize self so its exponent is the same as that of exp.

        Similar to self._rescale(exp._exp) but with error checking.
        """
        exp = _convert_other(exp, raiseit=True)
        if context is None:
            context = getcontext()
        if rounding is None:
            rounding = context.rounding
        if self._is_special or exp._is_special:
            ans = self._check_nans(exp, context)
            if ans:
                return ans
            if exp._isinfinity() or self._isinfinity():
                if exp._isinfinity() and self._isinfinity():
                    return Decimal(self)
                return context._raise_error(InvalidOperation, 'quantize with one INF')
        if not context.Etiny() <= exp._exp <= context.Emax:
            return context._raise_error(InvalidOperation, 'target exponent out of bounds in quantize')
        if not self:
            ans = _dec_from_triple(self._sign, '0', exp._exp)
            return ans._fix(context)
        self_adjusted = self.adjusted()
        if self_adjusted > context.Emax:
            return context._raise_error(InvalidOperation, 'exponent of quantize result too large for current context')
        if self_adjusted - exp._exp + 1 > context.prec:
            return context._raise_error(InvalidOperation, 'quantize result has too many digits for current context')
        ans = self._rescale(exp._exp, rounding)
        if ans.adjusted() > context.Emax:
            return context._raise_error(InvalidOperation, 'exponent of quantize result too large for current context')
        if len(ans._int) > context.prec:
            return context._raise_error(InvalidOperation, 'quantize result has too many digits for current context')
        if ans and ans.adjusted() < context.Emin:
            context._raise_error(Subnormal)
        if ans._exp > self._exp:
            if ans != self:
                context._raise_error(Inexact)
            context._raise_error(Rounded)
        ans = ans._fix(context)
        return ans

    def same_quantum(self, other, context=None):
        """Return True if self and other have the same exponent; otherwise
        return False.

        If either operand is a special value, the following rules are used:
           * return True if both operands are infinities
           * return True if both operands are NaNs
           * otherwise, return False.
        """
        other = _convert_other(other, raiseit=True)
        if self._is_special or other._is_special:
            return self.is_nan() and other.is_nan() or (self.is_infinite() and other.is_infinite())
        return self._exp == other._exp

    def _rescale(self, exp, rounding):
        """Rescale self so that the exponent is exp, either by padding with zeros
        or by truncating digits, using the given rounding mode.

        Specials are returned without change.  This operation is
        quiet: it raises no flags, and uses no information from the
        context.

        exp = exp to scale to (an integer)
        rounding = rounding mode
        """
        if self._is_special:
            return Decimal(self)
        if not self:
            return _dec_from_triple(self._sign, '0', exp)
        if self._exp >= exp:
            return _dec_from_triple(self._sign, self._int + '0' * (self._exp - exp), exp)
        digits = len(self._int) + self._exp - exp
        if digits < 0:
            self = _dec_from_triple(self._sign, '1', exp - 1)
            digits = 0
        this_function = self._pick_rounding_function[rounding]
        changed = this_function(self, digits)
        coeff = self._int[:digits] or '0'
        if changed == 1:
            coeff = str(int(coeff) + 1)
        return _dec_from_triple(self._sign, coeff, exp)

    def _round(self, places, rounding):
        """Round a nonzero, nonspecial Decimal to a fixed number of
        significant figures, using the given rounding mode.

        Infinities, NaNs and zeros are returned unaltered.

        This operation is quiet: it raises no flags, and uses no
        information from the context.

        """
        if places <= 0:
            raise ValueError('argument should be at least 1 in _round')
        if self._is_special or not self:
            return Decimal(self)
        ans = self._rescale(self.adjusted() + 1 - places, rounding)
        if ans.adjusted() != self.adjusted():
            ans = ans._rescale(ans.adjusted() + 1 - places, rounding)
        return ans

    def to_integral_exact(self, rounding=None, context=None):
        """Rounds to a nearby integer.

        If no rounding mode is specified, take the rounding mode from
        the context.  This method raises the Rounded and Inexact flags
        when appropriate.

        See also: to_integral_value, which does exactly the same as
        this method except that it doesn't raise Inexact or Rounded.
        """
        if self._is_special:
            ans = self._check_nans(context=context)
            if ans:
                return ans
            return Decimal(self)
        if self._exp >= 0:
            return Decimal(self)
        if not self:
            return _dec_from_triple(self._sign, '0', 0)
        if context is None:
            context = getcontext()
        if rounding is None:
            rounding = context.rounding
        ans = self._rescale(0, rounding)
        if ans != self:
            context._raise_error(Inexact)
        context._raise_error(Rounded)
        return ans

    def to_integral_value(self, rounding=None, context=None):
        """Rounds to the nearest integer, without raising inexact, rounded."""
        if context is None:
            context = getcontext()
        if rounding is None:
            rounding = context.rounding
        if self._is_special:
            ans = self._check_nans(context=context)
            if ans:
                return ans
            return Decimal(self)
        if self._exp >= 0:
            return Decimal(self)
        else:
            return self._rescale(0, rounding)
    to_integral = to_integral_value

    def sqrt(self, context=None):
        """Return the square root of self."""
        if context is None:
            context = getcontext()
        if self._is_special:
            ans = self._check_nans(context=context)
            if ans:
                return ans
            if self._isinfinity() and self._sign == 0:
                return Decimal(self)
        if not self:
            ans = _dec_from_triple(self._sign, '0', self._exp // 2)
            return ans._fix(context)
        if self._sign == 1:
            return context._raise_error(InvalidOperation, 'sqrt(-x), x > 0')
        prec = context.prec + 1
        op = _WorkRep(self)
        e = op.exp >> 1
        if op.exp & 1:
            c = op.int * 10
            l = (len(self._int) >> 1) + 1
        else:
            c = op.int
            l = len(self._int) + 1 >> 1
        shift = prec - l
        if shift >= 0:
            c *= 100 ** shift
            exact = True
        else:
            c, remainder = divmod(c, 100 ** (-shift))
            exact = not remainder
        e -= shift
        n = 10 ** prec
        while True:
            q = c // n
            if n <= q:
                break
            else:
                n = n + q >> 1
        exact = exact and n * n == c
        if exact:
            if shift >= 0:
                n //= 10 ** shift
            else:
                n *= 10 ** (-shift)
            e += shift
        elif n % 5 == 0:
            n += 1
        ans = _dec_from_triple(0, str(n), e)
        context = context._shallow_copy()
        rounding = context._set_rounding(ROUND_HALF_EVEN)
        ans = ans._fix(context)
        context.rounding = rounding
        return ans

    def max(self, other, context=None):
        """Returns the larger value.

        Like max(self, other) except if one is not a number, returns
        NaN (and signals if one is sNaN).  Also rounds.
        """
        other = _convert_other(other, raiseit=True)
        if context is None:
            context = getcontext()
        if self._is_special or other._is_special:
            sn = self._isnan()
            on = other._isnan()
            if sn or on:
                if on == 1 and sn == 0:
                    return self._fix(context)
                if sn == 1 and on == 0:
                    return other._fix(context)
                return self._check_nans(other, context)
        c = self._cmp(other)
        if c == 0:
            c = self.compare_total(other)
        if c == -1:
            ans = other
        else:
            ans = self
        return ans._fix(context)

    def min(self, other, context=None):
        """Returns the smaller value.

        Like min(self, other) except if one is not a number, returns
        NaN (and signals if one is sNaN).  Also rounds.
        """
        other = _convert_other(other, raiseit=True)
        if context is None:
            context = getcontext()
        if self._is_special or other._is_special:
            sn = self._isnan()
            on = other._isnan()
            if sn or on:
                if on == 1 and sn == 0:
                    return self._fix(context)
                if sn == 1 and on == 0:
                    return other._fix(context)
                return self._check_nans(other, context)
        c = self._cmp(other)
        if c == 0:
            c = self.compare_total(other)
        if c == -1:
            ans = self
        else:
            ans = other
        return ans._fix(context)

    def _isinteger(self):
        """Returns whether self is an integer"""
        if self._is_special:
            return False
        if self._exp >= 0:
            return True
        rest = self._int[self._exp:]
        return rest == '0' * len(rest)

    def _iseven(self):
        """Returns True if self is even.  Assumes self is an integer."""
        if not self or self._exp > 0:
            return True
        return self._int[-1 + self._exp] in '02468'

    def adjusted(self):
        """Return the adjusted exponent of self"""
        try:
            return self._exp + len(self._int) - 1
        except TypeError:
            return 0

    def canonical(self):
        """Returns the same Decimal object.

        As we do not have different encodings for the same number, the
        received object already is in its canonical form.
        """
        return self

    def compare_signal(self, other, context=None):
        """Compares self to the other operand numerically.

        It's pretty much like compare(), but all NaNs signal, with signaling
        NaNs taking precedence over quiet NaNs.
        """
        other = _convert_other(other, raiseit=True)
        ans = self._compare_check_nans(other, context)
        if ans:
            return ans
        return self.compare(other, context=context)

    def compare_total(self, other, context=None):
        """Compares self to other using the abstract representations.

        This is not like the standard compare, which use their numerical
        value. Note that a total ordering is defined for all possible abstract
        representations.
        """
        other = _convert_other(other, raiseit=True)
        if self._sign and (not other._sign):
            return _NegativeOne
        if not self._sign and other._sign:
            return _One
        sign = self._sign
        self_nan = self._isnan()
        other_nan = other._isnan()
        if self_nan or other_nan:
            if self_nan == other_nan:
                self_key = (len(self._int), self._int)
                other_key = (len(other._int), other._int)
                if self_key < other_key:
                    if sign:
                        return _One
                    else:
                        return _NegativeOne
                if self_key > other_key:
                    if sign:
                        return _NegativeOne
                    else:
                        return _One
                return _Zero
            if sign:
                if self_nan == 1:
                    return _NegativeOne
                if other_nan == 1:
                    return _One
                if self_nan == 2:
                    return _NegativeOne
                if other_nan == 2:
                    return _One
            else:
                if self_nan == 1:
                    return _One
                if other_nan == 1:
                    return _NegativeOne
                if self_nan == 2:
                    return _One
                if other_nan == 2:
                    return _NegativeOne
        if self < other:
            return _NegativeOne
        if self > other:
            return _One
        if self._exp < other._exp:
            if sign:
                return _One
            else:
                return _NegativeOne
        if self._exp > other._exp:
            if sign:
                return _NegativeOne
            else:
                return _One
        return _Zero

    def compare_total_mag(self, other, context=None):
        """Compares self to other using abstract repr., ignoring sign.

        Like compare_total, but with operand's sign ignored and assumed to be 0.
        """
        other = _convert_other(other, raiseit=True)
        s = self.copy_abs()
        o = other.copy_abs()
        return s.compare_total(o)

    def copy_abs(self):
        """Returns a copy with the sign set to 0. """
        return _dec_from_triple(0, self._int, self._exp, self._is_special)

    def copy_negate(self):
        """Returns a copy with the sign inverted."""
        if self._sign:
            return _dec_from_triple(0, self._int, self._exp, self._is_special)
        else:
            return _dec_from_triple(1, self._int, self._exp, self._is_special)

    def copy_sign(self, other, context=None):
        """Returns self with the sign of other."""
        other = _convert_other(other, raiseit=True)
        return _dec_from_triple(other._sign, self._int, self._exp, self._is_special)

    def exp(self, context=None):
        """Returns e ** self."""
        if context is None:
            context = getcontext()
        ans = self._check_nans(context=context)
        if ans:
            return ans
        if self._isinfinity() == -1:
            return _Zero
        if not self:
            return _One
        if self._isinfinity() == 1:
            return Decimal(self)
        p = context.prec
        adj = self.adjusted()
        if self._sign == 0 and adj > len(str((context.Emax + 1) * 3)):
            ans = _dec_from_triple(0, '1', context.Emax + 1)
        elif self._sign == 1 and adj > len(str((-context.Etiny() + 1) * 3)):
            ans = _dec_from_triple(0, '1', context.Etiny() - 1)
        elif self._sign == 0 and adj < -p:
            ans = _dec_from_triple(0, '1' + '0' * (p - 1) + '1', -p)
        elif self._sign == 1 and adj < -p - 1:
            ans = _dec_from_triple(0, '9' * (p + 1), -p - 1)
        else:
            op = _WorkRep(self)
            c, e = (op.int, op.exp)
            if op.sign == 1:
                c = -c
            extra = 3
            while True:
                coeff, exp = _dexp(c, e, p + extra)
                if coeff % (5 * 10 ** (len(str(coeff)) - p - 1)):
                    break
                extra += 3
            ans = _dec_from_triple(0, str(coeff), exp)
        context = context._shallow_copy()
        rounding = context._set_rounding(ROUND_HALF_EVEN)
        ans = ans._fix(context)
        context.rounding = rounding
        return ans

    def is_canonical(self):
        """Return True if self is canonical; otherwise return False.

        Currently, the encoding of a Decimal instance is always
        canonical, so this method returns True for any Decimal.
        """
        return True

    def is_finite(self):
        """Return True if self is finite; otherwise return False.

        A Decimal instance is considered finite if it is neither
        infinite nor a NaN.
        """
        return not self._is_special

    def is_infinite(self):
        """Return True if self is infinite; otherwise return False."""
        return self._exp == 'F'

    def is_nan(self):
        """Return True if self is a qNaN or sNaN; otherwise return False."""
        return self._exp in ('n', 'N')

    def is_normal(self, context=None):
        """Return True if self is a normal number; otherwise return False."""
        if self._is_special or not self:
            return False
        if context is None:
            context = getcontext()
        return context.Emin <= self.adjusted()

    def is_qnan(self):
        """Return True if self is a quiet NaN; otherwise return False."""
        return self._exp == 'n'

    def is_signed(self):
        """Return True if self is negative; otherwise return False."""
        return self._sign == 1

    def is_snan(self):
        """Return True if self is a signaling NaN; otherwise return False."""
        return self._exp == 'N'

    def is_subnormal(self, context=None):
        """Return True if self is subnormal; otherwise return False."""
        if self._is_special or not self:
            return False
        if context is None:
            context = getcontext()
        return self.adjusted() < context.Emin

    def is_zero(self):
        """Return True if self is a zero; otherwise return False."""
        return not self._is_special and self._int == '0'

    def _ln_exp_bound(self):
        """Compute a lower bound for the adjusted exponent of self.ln().
        In other words, compute r such that self.ln() >= 10**r.  Assumes
        that self is finite and positive and that self != 1.
        """
        adj = self._exp + len(self._int) - 1
        if adj >= 1:
            return len(str(adj * 23 // 10)) - 1
        if adj <= -2:
            return len(str((-1 - adj) * 23 // 10)) - 1
        op = _WorkRep(self)
        c, e = (op.int, op.exp)
        if adj == 0:
            num = str(c - 10 ** (-e))
            den = str(c)
            return len(num) - len(den) - (num < den)
        return e + len(str(10 ** (-e) - c)) - 1

    def ln(self, context=None):
        """Returns the natural (base e) logarithm of self."""
        if context is None:
            context = getcontext()
        ans = self._check_nans(context=context)
        if ans:
            return ans
        if not self:
            return _NegativeInfinity
        if self._isinfinity() == 1:
            return _Infinity
        if self == _One:
            return _Zero
        if self._sign == 1:
            return context._raise_error(InvalidOperation, 'ln of a negative value')
        op = _WorkRep(self)
        c, e = (op.int, op.exp)
        p = context.prec
        places = p - self._ln_exp_bound() + 2
        while True:
            coeff = _dlog(c, e, places)
            if coeff % (5 * 10 ** (len(str(abs(coeff))) - p - 1)):
                break
            places += 3
        ans = _dec_from_triple(int(coeff < 0), str(abs(coeff)), -places)
        context = context._shallow_copy()
        rounding = context._set_rounding(ROUND_HALF_EVEN)
        ans = ans._fix(context)
        context.rounding = rounding
        return ans

    def _log10_exp_bound(self):
        """Compute a lower bound for the adjusted exponent of self.log10().
        In other words, find r such that self.log10() >= 10**r.
        Assumes that self is finite and positive and that self != 1.
        """
        adj = self._exp + len(self._int) - 1
        if adj >= 1:
            return len(str(adj)) - 1
        if adj <= -2:
            return len(str(-1 - adj)) - 1
        op = _WorkRep(self)
        c, e = (op.int, op.exp)
        if adj == 0:
            num = str(c - 10 ** (-e))
            den = str(231 * c)
            return len(num) - len(den) - (num < den) + 2
        num = str(10 ** (-e) - c)
        return len(num) + e - (num < '231') - 1

    def log10(self, context=None):
        """Returns the base 10 logarithm of self."""
        if context is None:
            context = getcontext()
        ans = self._check_nans(context=context)
        if ans:
            return ans
        if not self:
            return _NegativeInfinity
        if self._isinfinity() == 1:
            return _Infinity
        if self._sign == 1:
            return context._raise_error(InvalidOperation, 'log10 of a negative value')
        if self._int[0] == '1' and self._int[1:] == '0' * (len(self._int) - 1):
            ans = Decimal(self._exp + len(self._int) - 1)
        else:
            op = _WorkRep(self)
            c, e = (op.int, op.exp)
            p = context.prec
            places = p - self._log10_exp_bound() + 2
            while True:
                coeff = _dlog10(c, e, places)
                if coeff % (5 * 10 ** (len(str(abs(coeff))) - p - 1)):
                    break
                places += 3
            ans = _dec_from_triple(int(coeff < 0), str(abs(coeff)), -places)
        context = context._shallow_copy()
        rounding = context._set_rounding(ROUND_HALF_EVEN)
        ans = ans._fix(context)
        context.rounding = rounding
        return ans

    def logb(self, context=None):
        """ Returns the exponent of the magnitude of self's MSD.

        The result is the integer which is the exponent of the magnitude
        of the most significant digit of self (as though it were truncated
        to a single digit while maintaining the value of that digit and
        without limiting the resulting exponent).
        """
        ans = self._check_nans(context=context)
        if ans:
            return ans
        if context is None:
            context = getcontext()
        if self._isinfinity():
            return _Infinity
        if not self:
            return context._raise_error(DivisionByZero, 'logb(0)', 1)
        ans = Decimal(self.adjusted())
        return ans._fix(context)

    def _islogical(self):
        """Return True if self is a logical operand.

        For being logical, it must be a finite number with a sign of 0,
        an exponent of 0, and a coefficient whose digits must all be
        either 0 or 1.
        """
        if self._sign != 0 or self._exp != 0:
            return False
        for dig in self._int:
            if dig not in '01':
                return False
        return True

    def _fill_logical(self, context, opa, opb):
        dif = context.prec - len(opa)
        if dif > 0:
            opa = '0' * dif + opa
        elif dif < 0:
            opa = opa[-context.prec:]
        dif = context.prec - len(opb)
        if dif > 0:
            opb = '0' * dif + opb
        elif dif < 0:
            opb = opb[-context.prec:]
        return (opa, opb)

    def logical_and(self, other, context=None):
        """Applies an 'and' operation between self and other's digits."""
        if context is None:
            context = getcontext()
        other = _convert_other(other, raiseit=True)
        if not self._islogical() or not other._islogical():
            return context._raise_error(InvalidOperation)
        opa, opb = self._fill_logical(context, self._int, other._int)
        result = ''.join([str(int(a) & int(b)) for a, b in zip(opa, opb)])
        return _dec_from_triple(0, result.lstrip('0') or '0', 0)

    def logical_invert(self, context=None):
        """Invert all its digits."""
        if context is None:
            context = getcontext()
        return self.logical_xor(_dec_from_triple(0, '1' * context.prec, 0), context)

    def logical_or(self, other, context=None):
        """Applies an 'or' operation between self and other's digits."""
        if context is None:
            context = getcontext()
        other = _convert_other(other, raiseit=True)
        if not self._islogical() or not other._islogical():
            return context._raise_error(InvalidOperation)
        opa, opb = self._fill_logical(context, self._int, other._int)
        result = ''.join([str(int(a) | int(b)) for a, b in zip(opa, opb)])
        return _dec_from_triple(0, result.lstrip('0') or '0', 0)

    def logical_xor(self, other, context=None):
        """Applies an 'xor' operation between self and other's digits."""
        if context is None:
            context = getcontext()
        other = _convert_other(other, raiseit=True)
        if not self._islogical() or not other._islogical():
            return context._raise_error(InvalidOperation)
        opa, opb = self._fill_logical(context, self._int, other._int)
        result = ''.join([str(int(a) ^ int(b)) for a, b in zip(opa, opb)])
        return _dec_from_triple(0, result.lstrip('0') or '0', 0)

    def max_mag(self, other, context=None):
        """Compares the values numerically with their sign ignored."""
        other = _convert_other(other, raiseit=True)
        if context is None:
            context = getcontext()
        if self._is_special or other._is_special:
            sn = self._isnan()
            on = other._isnan()
            if sn or on:
                if on == 1 and sn == 0:
                    return self._fix(context)
                if sn == 1 and on == 0:
                    return other._fix(context)
                return self._check_nans(other, context)
        c = self.copy_abs()._cmp(other.copy_abs())
        if c == 0:
            c = self.compare_total(other)
        if c == -1:
            ans = other
        else:
            ans = self
        return ans._fix(context)

    def min_mag(self, other, context=None):
        """Compares the values numerically with their sign ignored."""
        other = _convert_other(other, raiseit=True)
        if context is None:
            context = getcontext()
        if self._is_special or other._is_special:
            sn = self._isnan()
            on = other._isnan()
            if sn or on:
                if on == 1 and sn == 0:
                    return self._fix(context)
                if sn == 1 and on == 0:
                    return other._fix(context)
                return self._check_nans(other, context)
        c = self.copy_abs()._cmp(other.copy_abs())
        if c == 0:
            c = self.compare_total(other)
        if c == -1:
            ans = self
        else:
            ans = other
        return ans._fix(context)

    def next_minus(self, context=None):
        """Returns the largest representable number smaller than itself."""
        if context is None:
            context = getcontext()
        ans = self._check_nans(context=context)
        if ans:
            return ans
        if self._isinfinity() == -1:
            return _NegativeInfinity
        if self._isinfinity() == 1:
            return _dec_from_triple(0, '9' * context.prec, context.Etop())
        context = context.copy()
        context._set_rounding(ROUND_FLOOR)
        context._ignore_all_flags()
        new_self = self._fix(context)
        if new_self != self:
            return new_self
        return self.__sub__(_dec_from_triple(0, '1', context.Etiny() - 1), context)

    def next_plus(self, context=None):
        """Returns the smallest representable number larger than itself."""
        if context is None:
            context = getcontext()
        ans = self._check_nans(context=context)
        if ans:
            return ans
        if self._isinfinity() == 1:
            return _Infinity
        if self._isinfinity() == -1:
            return _dec_from_triple(1, '9' * context.prec, context.Etop())
        context = context.copy()
        context._set_rounding(ROUND_CEILING)
        context._ignore_all_flags()
        new_self = self._fix(context)
        if new_self != self:
            return new_self
        return self.__add__(_dec_from_triple(0, '1', context.Etiny() - 1), context)

    def next_toward(self, other, context=None):
        """Returns the number closest to self, in the direction towards other.

        The result is the closest representable number to self
        (excluding self) that is in the direction towards other,
        unless both have the same value.  If the two operands are
        numerically equal, then the result is a copy of self with the
        sign set to be the same as the sign of other.
        """
        other = _convert_other(other, raiseit=True)
        if context is None:
            context = getcontext()
        ans = self._check_nans(other, context)
        if ans:
            return ans
        comparison = self._cmp(other)
        if comparison == 0:
            return self.copy_sign(other)
        if comparison == -1:
            ans = self.next_plus(context)
        else:
            ans = self.next_minus(context)
        if ans._isinfinity():
            context._raise_error(Overflow, 'Infinite result from next_toward', ans._sign)
            context._raise_error(Inexact)
            context._raise_error(Rounded)
        elif ans.adjusted() < context.Emin:
            context._raise_error(Underflow)
            context._raise_error(Subnormal)
            context._raise_error(Inexact)
            context._raise_error(Rounded)
            if not ans:
                context._raise_error(Clamped)
        return ans

    def number_class(self, context=None):
        """Returns an indication of the class of self.

        The class is one of the following strings:
          sNaN
          NaN
          -Infinity
          -Normal
          -Subnormal
          -Zero
          +Zero
          +Subnormal
          +Normal
          +Infinity
        """
        if self.is_snan():
            return 'sNaN'
        if self.is_qnan():
            return 'NaN'
        inf = self._isinfinity()
        if inf == 1:
            return '+Infinity'
        if inf == -1:
            return '-Infinity'
        if self.is_zero():
            if self._sign:
                return '-Zero'
            else:
                return '+Zero'
        if context is None:
            context = getcontext()
        if self.is_subnormal(context=context):
            if self._sign:
                return '-Subnormal'
            else:
                return '+Subnormal'
        if self._sign:
            return '-Normal'
        else:
            return '+Normal'

    def radix(self):
        """Just returns 10, as this is Decimal, :)"""
        return Decimal(10)

    def rotate(self, other, context=None):
        """Returns a rotated copy of self, value-of-other times."""
        if context is None:
            context = getcontext()
        other = _convert_other(other, raiseit=True)
        ans = self._check_nans(other, context)
        if ans:
            return ans
        if other._exp != 0:
            return context._raise_error(InvalidOperation)
        if not -context.prec <= int(other) <= context.prec:
            return context._raise_error(InvalidOperation)
        if self._isinfinity():
            return Decimal(self)
        torot = int(other)
        rotdig = self._int
        topad = context.prec - len(rotdig)
        if topad > 0:
            rotdig = '0' * topad + rotdig
        elif topad < 0:
            rotdig = rotdig[-topad:]
        rotated = rotdig[torot:] + rotdig[:torot]
        return _dec_from_triple(self._sign, rotated.lstrip('0') or '0', self._exp)

    def scaleb(self, other, context=None):
        """Returns self operand after adding the second value to its exp."""
        if context is None:
            context = getcontext()
        other = _convert_other(other, raiseit=True)
        ans = self._check_nans(other, context)
        if ans:
            return ans
        if other._exp != 0:
            return context._raise_error(InvalidOperation)
        liminf = -2 * (context.Emax + context.prec)
        limsup = 2 * (context.Emax + context.prec)
        if not liminf <= int(other) <= limsup:
            return context._raise_error(InvalidOperation)
        if self._isinfinity():
            return Decimal(self)
        d = _dec_from_triple(self._sign, self._int, self._exp + int(other))
        d = d._fix(context)
        return d

    def shift(self, other, context=None):
        """Returns a shifted copy of self, value-of-other times."""
        if context is None:
            context = getcontext()
        other = _convert_other(other, raiseit=True)
        ans = self._check_nans(other, context)
        if ans:
            return ans
        if other._exp != 0:
            return context._raise_error(InvalidOperation)
        if not -context.prec <= int(other) <= context.prec:
            return context._raise_error(InvalidOperation)
        if self._isinfinity():
            return Decimal(self)
        torot = int(other)
        rotdig = self._int
        topad = context.prec - len(rotdig)
        if topad > 0:
            rotdig = '0' * topad + rotdig
        elif topad < 0:
            rotdig = rotdig[-topad:]
        if torot < 0:
            shifted = rotdig[:torot]
        else:
            shifted = rotdig + '0' * torot
            shifted = shifted[-context.prec:]
        return _dec_from_triple(self._sign, shifted.lstrip('0') or '0', self._exp)

    def __reduce__(self):
        return (self.__class__, (str(self),))

    def __copy__(self):
        if type(self) is Decimal:
            return self
        return self.__class__(str(self))

    def __deepcopy__(self, memo):
        if type(self) is Decimal:
            return self
        return self.__class__(str(self))

    def __format__(self, specifier, context=None, _localeconv=None):
        """Format a Decimal instance according to the given specifier.

        The specifier should be a standard format specifier, with the
        form described in PEP 3101.  Formatting types 'e', 'E', 'f',
        'F', 'g', 'G', 'n' and '%' are supported.  If the formatting
        type is omitted it defaults to 'g' or 'G', depending on the
        value of context.capitals.
        """
        if context is None:
            context = getcontext()
        spec = _parse_format_specifier(specifier, _localeconv=_localeconv)
        if self._is_special:
            sign = _format_sign(self._sign, spec)
            body = str(self.copy_abs())
            if spec['type'] == '%':
                body += '%'
            return _format_align(sign, body, spec)
        if spec['type'] is None:
            spec['type'] = ['g', 'G'][context.capitals]
        if spec['type'] == '%':
            self = _dec_from_triple(self._sign, self._int, self._exp + 2)
        rounding = context.rounding
        precision = spec['precision']
        if precision is not None:
            if spec['type'] in 'eE':
                self = self._round(precision + 1, rounding)
            elif spec['type'] in 'fF%':
                self = self._rescale(-precision, rounding)
            elif spec['type'] in 'gG' and len(self._int) > precision:
                self = self._round(precision, rounding)
        if not self and self._exp > 0 and (spec['type'] in 'fF%'):
            self = self._rescale(0, rounding)
        if not self and spec['no_neg_0'] and self._sign:
            adjusted_sign = 0
        else:
            adjusted_sign = self._sign
        leftdigits = self._exp + len(self._int)
        if spec['type'] in 'eE':
            if not self and precision is not None:
                dotplace = 1 - precision
            else:
                dotplace = 1
        elif spec['type'] in 'fF%':
            dotplace = leftdigits
        elif spec['type'] in 'gG':
            if self._exp <= 0 and leftdigits > -6:
                dotplace = leftdigits
            else:
                dotplace = 1
        if dotplace < 0:
            intpart = '0'
            fracpart = '0' * -dotplace + self._int
        elif dotplace > len(self._int):
            intpart = self._int + '0' * (dotplace - len(self._int))
            fracpart = ''
        else:
            intpart = self._int[:dotplace] or '0'
            fracpart = self._int[dotplace:]
        exp = leftdigits - dotplace
        return _format_number(adjusted_sign, intpart, fracpart, exp, spec)