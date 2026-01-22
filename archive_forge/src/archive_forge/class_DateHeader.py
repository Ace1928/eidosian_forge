from types import MappingProxyType
from email import utils
from email import errors
from email import _header_value_parser as parser
class DateHeader:
    """Header whose value consists of a single timestamp.

    Provides an additional attribute, datetime, which is either an aware
    datetime using a timezone, or a naive datetime if the timezone
    in the input string is -0000.  Also accepts a datetime as input.
    The 'value' attribute is the normalized form of the timestamp,
    which means it is the output of format_datetime on the datetime.
    """
    max_count = None
    value_parser = staticmethod(parser.get_unstructured)

    @classmethod
    def parse(cls, value, kwds):
        if not value:
            kwds['defects'].append(errors.HeaderMissingRequiredValue())
            kwds['datetime'] = None
            kwds['decoded'] = ''
            kwds['parse_tree'] = parser.TokenList()
            return
        if isinstance(value, str):
            kwds['decoded'] = value
            try:
                value = utils.parsedate_to_datetime(value)
            except ValueError:
                kwds['defects'].append(errors.InvalidDateDefect('Invalid date value or format'))
                kwds['datetime'] = None
                kwds['parse_tree'] = parser.TokenList()
                return
        kwds['datetime'] = value
        kwds['decoded'] = utils.format_datetime(kwds['datetime'])
        kwds['parse_tree'] = cls.value_parser(kwds['decoded'])

    def init(self, *args, **kw):
        self._datetime = kw.pop('datetime')
        super().init(*args, **kw)

    @property
    def datetime(self):
        return self._datetime