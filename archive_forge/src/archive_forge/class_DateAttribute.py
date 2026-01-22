import re
import datetime
import numpy as np
import csv
import ctypes
class DateAttribute(Attribute):

    def __init__(self, name, date_format, datetime_unit):
        super().__init__(name)
        self.date_format = date_format
        self.datetime_unit = datetime_unit
        self.type_name = 'date'
        self.range = date_format
        self.dtype = np.datetime64(0, self.datetime_unit)

    @staticmethod
    def _get_date_format(atrv):
        m = r_date.match(atrv)
        if m:
            pattern = m.group(1).strip()
            datetime_unit = None
            if 'yyyy' in pattern:
                pattern = pattern.replace('yyyy', '%Y')
                datetime_unit = 'Y'
            elif 'yy':
                pattern = pattern.replace('yy', '%y')
                datetime_unit = 'Y'
            if 'MM' in pattern:
                pattern = pattern.replace('MM', '%m')
                datetime_unit = 'M'
            if 'dd' in pattern:
                pattern = pattern.replace('dd', '%d')
                datetime_unit = 'D'
            if 'HH' in pattern:
                pattern = pattern.replace('HH', '%H')
                datetime_unit = 'h'
            if 'mm' in pattern:
                pattern = pattern.replace('mm', '%M')
                datetime_unit = 'm'
            if 'ss' in pattern:
                pattern = pattern.replace('ss', '%S')
                datetime_unit = 's'
            if 'z' in pattern or 'Z' in pattern:
                raise ValueError('Date type attributes with time zone not supported, yet')
            if datetime_unit is None:
                raise ValueError('Invalid or unsupported date format')
            return (pattern, datetime_unit)
        else:
            raise ValueError('Invalid or no date format')

    @classmethod
    def parse_attribute(cls, name, attr_string):
        """
        Parse the attribute line if it knows how. Returns the parsed
        attribute, or None.

        For date attributes, the attribute string would be like
        'date <format>'.
        """
        attr_string_lower = attr_string.lower().strip()
        if attr_string_lower[:len('date')] == 'date':
            date_format, datetime_unit = cls._get_date_format(attr_string)
            return cls(name, date_format, datetime_unit)
        else:
            return None

    def parse_data(self, data_str):
        """
        Parse a value of this type.
        """
        date_str = data_str.strip().strip("'").strip('"')
        if date_str == '?':
            return np.datetime64('NaT', self.datetime_unit)
        else:
            dt = datetime.datetime.strptime(date_str, self.date_format)
            return np.datetime64(dt).astype('datetime64[%s]' % self.datetime_unit)

    def __str__(self):
        return super().__str__() + ',' + self.date_format