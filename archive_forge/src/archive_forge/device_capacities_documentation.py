from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
Generate a list of devices/OS versions & corresponding capacity info.

    Args:
      catalog: Android or iOS catalog

    Returns:
      The list of device models, versions, and capacity info we want to have
      printed later. Obsolete (unsupported) devices, versions, and entries
      missing capacity info are filtered out.
    