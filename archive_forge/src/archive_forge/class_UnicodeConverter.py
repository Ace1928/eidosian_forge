from __future__ import annotations
import re
import typing as t
import uuid
from urllib.parse import quote
class UnicodeConverter(BaseConverter):
    """This converter is the default converter and accepts any string but
    only one path segment.  Thus the string can not include a slash.

    This is the default validator.

    Example::

        Rule('/pages/<page>'),
        Rule('/<string(length=2):lang_code>')

    :param map: the :class:`Map`.
    :param minlength: the minimum length of the string.  Must be greater
                      or equal 1.
    :param maxlength: the maximum length of the string.
    :param length: the exact length of the string.
    """

    def __init__(self, map: Map, minlength: int=1, maxlength: int | None=None, length: int | None=None) -> None:
        super().__init__(map)
        if length is not None:
            length_regex = f'{{{int(length)}}}'
        else:
            if maxlength is None:
                maxlength_value = ''
            else:
                maxlength_value = str(int(maxlength))
            length_regex = f'{{{int(minlength)},{maxlength_value}}}'
        self.regex = f'[^/]{length_regex}'