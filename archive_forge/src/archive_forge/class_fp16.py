import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class fp16(Stub):
    """Namespace for fp16 operations
    """
    _description_ = '<fp16>'

    class hadd(Stub):
        """hadd(a, b)

        Perform fp16 addition, (a + b) in round to nearest mode. Supported
        on fp16 operands only.

        Returns the fp16 result of the addition.

        """

    class hsub(Stub):
        """hsub(a, b)

        Perform fp16 subtraction, (a - b) in round to nearest mode. Supported
        on fp16 operands only.

        Returns the fp16 result of the subtraction.

        """

    class hmul(Stub):
        """hmul(a, b)

        Perform fp16 multiplication, (a * b) in round to nearest mode. Supported
        on fp16 operands only.

        Returns the fp16 result of the multiplication.

        """

    class hdiv(Stub):
        """hdiv(a, b)

        Perform fp16 division, (a / b) in round to nearest mode. Supported
        on fp16 operands only.

        Returns the fp16 result of the division

        """

    class hfma(Stub):
        """hfma(a, b, c)

        Perform fp16 multiply and accumulate, (a * b) + c in round to nearest
        mode. Supported on fp16 operands only.

        Returns the fp16 result of the multiplication.

        """

    class hneg(Stub):
        """hneg(a)

        Perform fp16 negation, -(a). Supported on fp16 operands only.

        Returns the fp16 result of the negation.

        """

    class habs(Stub):
        """habs(a)

        Perform fp16 absolute value, |a|. Supported on fp16 operands only.

        Returns the fp16 result of the absolute value.

        """

    class hsin(Stub):
        """hsin(a)

        Calculate sine in round to nearest even mode. Supported on fp16
        operands only.

        Returns the sine result.

        """

    class hcos(Stub):
        """hsin(a)

        Calculate cosine in round to nearest even mode. Supported on fp16
        operands only.

        Returns the cosine result.

        """

    class hlog(Stub):
        """hlog(a)

        Calculate natural logarithm in round to nearest even mode. Supported
        on fp16 operands only.

        Returns the natural logarithm result.

        """

    class hlog10(Stub):
        """hlog10(a)

        Calculate logarithm base 10 in round to nearest even mode. Supported
        on fp16 operands only.

        Returns the logarithm base 10 result.

        """

    class hlog2(Stub):
        """hlog2(a)

        Calculate logarithm base 2 in round to nearest even mode. Supported
        on fp16 operands only.

        Returns the logarithm base 2 result.

        """

    class hexp(Stub):
        """hexp(a)

        Calculate natural exponential, exp(a), in round to nearest mode.
        Supported on fp16 operands only.

        Returns the natural exponential result.

        """

    class hexp10(Stub):
        """hexp10(a)

        Calculate exponential base 10 (10 ** a) in round to nearest mode.
        Supported on fp16 operands only.

        Returns the exponential base 10 result.

        """

    class hexp2(Stub):
        """hexp2(a)

        Calculate exponential base 2 (2 ** a) in round to nearest mode.
        Supported on fp16 operands only.

        Returns the exponential base 2 result.

        """

    class hfloor(Stub):
        """hfloor(a)

        Calculate the floor, the largest integer less than or equal to 'a'.
        Supported on fp16 operands only.

        Returns the floor result.

        """

    class hceil(Stub):
        """hceil(a)

        Calculate the ceil, the smallest integer greater than or equal to 'a'.
        Supported on fp16 operands only.

        Returns the ceil result.

        """

    class hsqrt(Stub):
        """hsqrt(a)

        Calculate the square root of the input argument in round to nearest
        mode. Supported on fp16 operands only.

        Returns the square root result.

        """

    class hrsqrt(Stub):
        """hrsqrt(a)

        Calculate the reciprocal square root of the input argument in round
        to nearest even mode. Supported on fp16 operands only.

        Returns the reciprocal square root result.

        """

    class hrcp(Stub):
        """hrcp(a)

        Calculate the reciprocal of the input argument in round to nearest
        even mode. Supported on fp16 operands only.

        Returns the reciprocal result.

        """

    class hrint(Stub):
        """hrint(a)

        Round the input argument to nearest integer value. Supported on fp16
        operands only.

        Returns the rounded result.

        """

    class htrunc(Stub):
        """htrunc(a)

        Truncate the input argument to its integer portion. Supported
        on fp16 operands only.

        Returns the truncated result.

        """

    class heq(Stub):
        """heq(a, b)

        Perform fp16 comparison, (a == b). Supported
        on fp16 operands only.

        Returns True if a and b are equal and False otherwise.

        """

    class hne(Stub):
        """hne(a, b)

        Perform fp16 comparison, (a != b). Supported
        on fp16 operands only.

        Returns True if a and b are not equal and False otherwise.

        """

    class hge(Stub):
        """hge(a, b)

        Perform fp16 comparison, (a >= b). Supported
        on fp16 operands only.

        Returns True if a is >= b and False otherwise.

        """

    class hgt(Stub):
        """hgt(a, b)

        Perform fp16 comparison, (a > b). Supported
        on fp16 operands only.

        Returns True if a is > b and False otherwise.

        """

    class hle(Stub):
        """hle(a, b)

        Perform fp16 comparison, (a <= b). Supported
        on fp16 operands only.

        Returns True if a is <= b and False otherwise.

        """

    class hlt(Stub):
        """hlt(a, b)

        Perform fp16 comparison, (a < b). Supported
        on fp16 operands only.

        Returns True if a is < b and False otherwise.

        """

    class hmax(Stub):
        """hmax(a, b)

        Perform fp16 maximum operation, max(a,b) Supported
        on fp16 operands only.

        Returns a if a is greater than b, returns b otherwise.

        """

    class hmin(Stub):
        """hmin(a, b)

        Perform fp16 minimum operation, min(a,b). Supported
        on fp16 operands only.

        Returns a if a is less than b, returns b otherwise.

        """