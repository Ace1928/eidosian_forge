from __future__ import absolute_import, division, print_function
import abc
import binascii
import os
from base64 import b64encode
from datetime import datetime
from hashlib import sha256
from ansible.module_utils import six
from ansible.module_utils.common.text.converters import to_text
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import convert_relative_to_datetime
from ansible_collections.community.crypto.plugins.module_utils.openssh.utils import (
class OpensshCertificateTimeParameters(object):

    def __init__(self, valid_from, valid_to):
        self._valid_from = self.to_datetime(valid_from)
        self._valid_to = self.to_datetime(valid_to)
        if self._valid_from > self._valid_to:
            raise ValueError('Valid from: %s must not be greater than Valid to: %s' % (valid_from, valid_to))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        else:
            return self._valid_from == other._valid_from and self._valid_to == other._valid_to

    def __ne__(self, other):
        return not self == other

    @property
    def validity_string(self):
        if not (self._valid_from == _ALWAYS and self._valid_to == _FOREVER):
            return '%s:%s' % (self.valid_from(date_format='openssh'), self.valid_to(date_format='openssh'))
        return ''

    def valid_from(self, date_format):
        return self.format_datetime(self._valid_from, date_format)

    def valid_to(self, date_format):
        return self.format_datetime(self._valid_to, date_format)

    def within_range(self, valid_at):
        if valid_at is not None:
            valid_at_datetime = self.to_datetime(valid_at)
            return self._valid_from <= valid_at_datetime <= self._valid_to
        return True

    @staticmethod
    def format_datetime(dt, date_format):
        if date_format in ('human_readable', 'openssh'):
            if dt == _ALWAYS:
                result = 'always'
            elif dt == _FOREVER:
                result = 'forever'
            else:
                result = dt.isoformat() if date_format == 'human_readable' else dt.strftime('%Y%m%d%H%M%S')
        elif date_format == 'timestamp':
            td = dt - _ALWAYS
            result = int((td.microseconds + (td.seconds + td.days * 24 * 3600) * 10 ** 6) / 10 ** 6)
        else:
            raise ValueError('%s is not a valid format' % date_format)
        return result

    @staticmethod
    def to_datetime(time_string_or_timestamp):
        try:
            if isinstance(time_string_or_timestamp, six.string_types):
                result = OpensshCertificateTimeParameters._time_string_to_datetime(time_string_or_timestamp.strip())
            elif isinstance(time_string_or_timestamp, (long, int)):
                result = OpensshCertificateTimeParameters._timestamp_to_datetime(time_string_or_timestamp)
            else:
                raise ValueError('Value must be of type (str, unicode, int, long) not %s' % type(time_string_or_timestamp))
        except ValueError:
            raise
        return result

    @staticmethod
    def _timestamp_to_datetime(timestamp):
        if timestamp == 0:
            result = _ALWAYS
        elif timestamp == 18446744073709551615:
            result = _FOREVER
        else:
            try:
                result = datetime.utcfromtimestamp(timestamp)
            except OverflowError as e:
                raise ValueError
        return result

    @staticmethod
    def _time_string_to_datetime(time_string):
        result = None
        if time_string == 'always':
            result = _ALWAYS
        elif time_string == 'forever':
            result = _FOREVER
        elif is_relative_time_string(time_string):
            result = convert_relative_to_datetime(time_string)
        else:
            for time_format in ('%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S'):
                try:
                    result = datetime.strptime(time_string, time_format)
                except ValueError:
                    pass
            if result is None:
                raise ValueError
        return result