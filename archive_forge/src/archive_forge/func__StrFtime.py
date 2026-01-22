from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import re
from dateutil import parser
from dateutil import tz
from dateutil.tz import _common as tz_common
import enum
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import iso_duration
from googlecloudsdk.core.util import times_data
import six
def _StrFtime(dt, fmt):
    """Convert strftime exceptions to Datetime Errors."""
    try:
        return dt.strftime(fmt)
    except (TypeError, UnicodeError) as e:
        if '%Z' not in fmt:
            raise DateTimeValueError(six.text_type(e))
        return FormatDateTime(dt, fmt.replace('%Z', '%Ez'))
    except (AttributeError, OverflowError, ValueError) as e:
        raise DateTimeValueError(six.text_type(e))