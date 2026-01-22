import collections
import copy
import hashlib
import json
import multiprocessing
import os.path
import re
import signal
import subprocess
import sys
import gyp
import gyp.common
import gyp.msvs_emulation
import gyp.MSVSUtil as MSVSUtil
import gyp.xcode_emulation
from io import StringIO
from gyp.common import GetEnvironFallback
import gyp.ninja_syntax as ninja_syntax
def ComputeOutputFileName(self, spec, type=None):
    """Compute the filename of the final output for the current target."""
    if not type:
        type = spec['type']
    default_variables = copy.copy(generator_default_variables)
    CalculateVariables(default_variables, {'flavor': self.flavor})
    DEFAULT_PREFIX = {'loadable_module': default_variables['SHARED_LIB_PREFIX'], 'shared_library': default_variables['SHARED_LIB_PREFIX'], 'static_library': default_variables['STATIC_LIB_PREFIX'], 'executable': default_variables['EXECUTABLE_PREFIX']}
    prefix = spec.get('product_prefix', DEFAULT_PREFIX.get(type, ''))
    DEFAULT_EXTENSION = {'loadable_module': default_variables['SHARED_LIB_SUFFIX'], 'shared_library': default_variables['SHARED_LIB_SUFFIX'], 'static_library': default_variables['STATIC_LIB_SUFFIX'], 'executable': default_variables['EXECUTABLE_SUFFIX']}
    extension = spec.get('product_extension')
    if extension:
        extension = '.' + extension
    else:
        extension = DEFAULT_EXTENSION.get(type, '')
    if 'product_name' in spec:
        target = spec['product_name']
    else:
        target = spec['target_name']
        if prefix == 'lib':
            target = StripPrefix(target, 'lib')
    if type in ('static_library', 'loadable_module', 'shared_library', 'executable'):
        return f'{prefix}{target}{extension}'
    elif type == 'none':
        return '%s.stamp' % target
    else:
        raise Exception('Unhandled output type %s' % type)