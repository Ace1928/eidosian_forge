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
def _render_module_flags(self, module, flags, output_lines, prefix=''):
    """Returns a help string for a given module."""
    if not isinstance(module, str):
        module = module.__name__
    output_lines.append('\n%s%s:' % (prefix, module))
    self._render_flag_list(flags, output_lines, prefix + '  ')