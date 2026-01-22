import re, textwrap, os
from os import sys, path
from distutils.errors import DistutilsError
def arg_regex(self, **kwargs):
    map2origin = dict(x64='x86', ppc64le='ppc64', aarch64='armhf', clang='gcc')
    march = self.march()
    cc_name = self.cc_name()
    map_march = map2origin.get(march, march)
    map_cc = map2origin.get(cc_name, cc_name)
    for key in (march, cc_name, map_march, map_cc, march + '_' + cc_name, map_march + '_' + cc_name, march + '_' + map_cc, map_march + '_' + map_cc):
        regex = kwargs.pop(key, None)
        if regex is not None:
            break
    if regex:
        if isinstance(regex, dict):
            for k, v in regex.items():
                if v[-1:] not in ')}$?\\.+*':
                    regex[k] = v + '$'
        else:
            assert isinstance(regex, str)
            if regex[-1:] not in ')}$?\\.+*':
                regex += '$'
    return regex