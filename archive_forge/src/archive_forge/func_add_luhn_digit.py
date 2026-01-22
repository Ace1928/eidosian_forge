import contextlib
import datetime
import ipaddress
import json
import math
from fractions import Fraction
from typing import Callable, Dict, Type, Union, cast, overload
import hypothesis.strategies as st
import pydantic
import pydantic.color
import pydantic.types
from pydantic.utils import lenient_issubclass
def add_luhn_digit(card_number: str) -> str:
    for digit in '0123456789':
        with contextlib.suppress(Exception):
            pydantic.PaymentCardNumber.validate_luhn_check_digit(card_number + digit)
            return card_number + digit
    raise AssertionError('Unreachable')