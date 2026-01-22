from collections import namedtuple
import re
import textwrap
import warnings
class AcceptCharset(object):
    """
    Represent an ``Accept-Charset`` header.

    Base class for :class:`AcceptCharsetValidHeader`,
    :class:`AcceptCharsetNoHeader`, and :class:`AcceptCharsetInvalidHeader`.
    """
    charset_re = token_re
    charset_n_weight_re = _item_n_weight_re(item_re=charset_re)
    charset_n_weight_compiled_re = re.compile(charset_n_weight_re)
    accept_charset_compiled_re = _list_1_or_more__compiled_re(element_re=charset_n_weight_re)

    @classmethod
    def _python_value_to_header_str(cls, value):
        if isinstance(value, str):
            header_str = value
        else:
            if hasattr(value, 'items'):
                value = sorted(value.items(), key=lambda item: item[1], reverse=True)
            if isinstance(value, (tuple, list)):
                result = []
                for item in value:
                    if isinstance(item, (tuple, list)):
                        item = _item_qvalue_pair_to_header_element(pair=item)
                    result.append(item)
                header_str = ', '.join(result)
            else:
                header_str = str(value)
        return header_str

    @classmethod
    def parse(cls, value):
        """
        Parse an ``Accept-Charset`` header.

        :param value: (``str``) header value
        :return: If `value` is a valid ``Accept-Charset`` header, returns an
                 iterator of (charset, quality value) tuples, as parsed from
                 the header from left to right.
        :raises ValueError: if `value` is an invalid header
        """
        if cls.accept_charset_compiled_re.match(value) is None:
            raise ValueError('Invalid value for an Accept-Charset header.')

        def generator(value):
            for match in cls.charset_n_weight_compiled_re.finditer(value):
                charset = match.group(1)
                qvalue = match.group(2)
                qvalue = float(qvalue) if qvalue else 1.0
                yield (charset, qvalue)
        return generator(value=value)