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
def _check_method_name_conflicts(self, name, flag):
    if flag.allow_using_method_names:
        return
    short_name = flag.short_name
    flag_names = {name} if short_name is None else {name, short_name}
    for flag_name in flag_names:
        if flag_name in self.__dict__['__banned_flag_names']:
            raise _exceptions.FlagNameConflictsWithMethodError('Cannot define a flag named "{name}". It conflicts with a method on class "{class_name}". To allow defining it, use allow_using_method_names and access the flag value with FLAGS[\'{name}\'].value. FLAGS.{name} returns the method, not the flag value.'.format(name=flag_name, class_name=type(self).__name__))