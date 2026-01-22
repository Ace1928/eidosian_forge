from collections import namedtuple
import re
import textwrap
import warnings
class AcceptLanguage(object):
    """
    Represent an ``Accept-Language`` header.

    Base class for :class:`AcceptLanguageValidHeader`,
    :class:`AcceptLanguageNoHeader`, and :class:`AcceptLanguageInvalidHeader`.
    """
    lang_range_re = '\\*|(?:[A-Za-z]{1,8}(?:-[A-Za-z0-9]{1,8})*)'
    lang_range_n_weight_re = _item_n_weight_re(item_re=lang_range_re)
    lang_range_n_weight_compiled_re = re.compile(lang_range_n_weight_re)
    accept_language_compiled_re = _list_1_or_more__compiled_re(element_re=lang_range_n_weight_re)

    @classmethod
    def _python_value_to_header_str(cls, value):
        if isinstance(value, str):
            header_str = value
        else:
            if hasattr(value, 'items'):
                value = sorted(value.items(), key=lambda item: item[1], reverse=True)
            if isinstance(value, (tuple, list)):
                result = []
                for element in value:
                    if isinstance(element, (tuple, list)):
                        element = _item_qvalue_pair_to_header_element(pair=element)
                    result.append(element)
                header_str = ', '.join(result)
            else:
                header_str = str(value)
        return header_str

    @classmethod
    def parse(cls, value):
        """
        Parse an ``Accept-Language`` header.

        :param value: (``str``) header value
        :return: If `value` is a valid ``Accept-Language`` header, returns an
                 iterator of (language range, quality value) tuples, as parsed
                 from the header from left to right.
        :raises ValueError: if `value` is an invalid header
        """
        if cls.accept_language_compiled_re.match(value) is None:
            raise ValueError('Invalid value for an Accept-Language header.')

        def generator(value):
            for match in cls.lang_range_n_weight_compiled_re.finditer(value):
                lang_range = match.group(1)
                qvalue = match.group(2)
                qvalue = float(qvalue) if qvalue else 1.0
                yield (lang_range, qvalue)
        return generator(value=value)