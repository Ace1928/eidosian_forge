from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from datetime import timedelta
from datetime import tzinfo
A fixed offset to handle daylight savings time (DST).

    Returns:
      A time duration of zero as UTC does not use DST.
    