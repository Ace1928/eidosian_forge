from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import itertools
import logging
import os
import sys
from xml.dom import minidom
from absl.flags import _exceptions
from absl.flags import _flag
from absl.flags import _helpers
from absl.flags import _validators_classes
import six
def _flag_is_registered(self, flag_obj):
    """Checks whether a Flag object is registered under long name or short name.

    Args:
      flag_obj: Flag, the Flag instance to check for.

    Returns:
      bool, True iff flag_obj is registered under long name or short name.
    """
    flag_dict = self._flags()
    name = flag_obj.name
    if flag_dict.get(name, None) == flag_obj:
        return True
    short_name = flag_obj.short_name
    if short_name is not None and flag_dict.get(short_name, None) == flag_obj:
        return True
    return False