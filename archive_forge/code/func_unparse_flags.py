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
def unparse_flags(self):
    """Unparses all flags to the point before any FLAGS(argv) was called."""
    for f in self._flags().values():
        f.unparse()
    logging.info('unparse_flags() called; flags access will now raise errors.')
    self.__dict__['__flags_parsed'] = False
    self.__dict__['__unparse_flags_called'] = True