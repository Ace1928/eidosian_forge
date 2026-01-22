from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class NumberLocale(VegaLiteSchema):
    """NumberLocale schema wrapper
    Locale definition for formatting numbers.

    Parameters
    ----------

    currency : Sequence[str], :class:`Vector2string`
        The currency prefix and suffix (e.g., ["$", ""]).
    decimal : str
        The decimal point (e.g., ".").
    grouping : Sequence[float]
        The array of group sizes (e.g., [3]), cycled as needed.
    thousands : str
        The group separator (e.g., ",").
    minus : str
        The minus sign (defaults to hyphen-minus, "-").
    nan : str
        The not-a-number value (defaults to "NaN").
    numerals : Sequence[str], :class:`Vector10string`
        An array of ten strings to replace the numerals 0-9.
    percent : str
        The percent sign (defaults to "%").
    """
    _schema = {'$ref': '#/definitions/NumberLocale'}

    def __init__(self, currency: Union['SchemaBase', Sequence[str], UndefinedType]=Undefined, decimal: Union[str, UndefinedType]=Undefined, grouping: Union[Sequence[float], UndefinedType]=Undefined, thousands: Union[str, UndefinedType]=Undefined, minus: Union[str, UndefinedType]=Undefined, nan: Union[str, UndefinedType]=Undefined, numerals: Union['SchemaBase', Sequence[str], UndefinedType]=Undefined, percent: Union[str, UndefinedType]=Undefined, **kwds):
        super(NumberLocale, self).__init__(currency=currency, decimal=decimal, grouping=grouping, thousands=thousands, minus=minus, nan=nan, numerals=numerals, percent=percent, **kwds)