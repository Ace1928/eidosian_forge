import sys
from distutils.core import Distribution
import warnings
import distutils.core
import distutils.dist
from numpy.distutils.extension import Extension  # noqa: F401
from numpy.distutils.numpy_distribution import NumpyDistribution
from numpy.distutils.command import config, config_compiler, \
from numpy.distutils.misc_util import is_sequence, is_string
def _check_append_library(libraries, item):
    for libitem in libraries:
        if is_sequence(libitem):
            if is_sequence(item):
                if item[0] == libitem[0]:
                    if item[1] is libitem[1]:
                        return
                    warnings.warn('[0] libraries list contains %r with different build_info' % (item[0],), stacklevel=2)
                    break
            elif item == libitem[0]:
                warnings.warn('[1] libraries list contains %r with no build_info' % (item[0],), stacklevel=2)
                break
        elif is_sequence(item):
            if item[0] == libitem:
                warnings.warn('[2] libraries list contains %r with no build_info' % (item[0],), stacklevel=2)
                break
        elif item == libitem:
            return
    libraries.append(item)