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
def _set_attributes(self, **attributes):
    """Sets multiple flag values together, triggers validators afterwards."""
    fl = self._flags()
    known_flags = set()
    for name, value in six.iteritems(attributes):
        if name in self.__dict__['__hiddenflags']:
            raise AttributeError(name)
        if name in fl:
            fl[name].value = value
            known_flags.add(name)
        else:
            self._set_unknown_flag(name, value)
    for name in known_flags:
        self._assert_validators(fl[name].validators)
        fl[name].using_default_value = False