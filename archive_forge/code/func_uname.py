import collections
import os
import re
import sys
import functools
import itertools
def uname():
    """ Fairly portable uname interface. Returns a tuple
        of strings (system, node, release, version, machine, processor)
        identifying the underlying platform.

        Note that unlike the os.uname function this also returns
        possible processor information as an additional tuple entry.

        Entries which cannot be determined are set to ''.

    """
    global _uname_cache
    if _uname_cache is not None:
        return _uname_cache
    try:
        system, node, release, version, machine = infos = os.uname()
    except AttributeError:
        system = sys.platform
        node = _node()
        release = version = machine = ''
        infos = ()
    if not any(infos):
        if system == 'win32':
            release, version, csd, ptype = win32_ver()
            machine = machine or _get_machine_win32()
        if not (release and version):
            system, release, version = _syscmd_ver(system)
            if system == 'Microsoft Windows':
                system = 'Windows'
            elif system == 'Microsoft' and release == 'Windows':
                system = 'Windows'
                if '6.0' == version[:3]:
                    release = 'Vista'
                else:
                    release = ''
        if system in ('win32', 'win16'):
            if not version:
                if system == 'win32':
                    version = '32bit'
                else:
                    version = '16bit'
            system = 'Windows'
        elif system[:4] == 'java':
            release, vendor, vminfo, osinfo = java_ver()
            system = 'Java'
            version = ', '.join(vminfo)
            if not version:
                version = vendor
    if system == 'OpenVMS':
        if not release or release == '0':
            release = version
            version = ''
    if system == 'Microsoft' and release == 'Windows':
        system = 'Windows'
        release = 'Vista'
    vals = (system, node, release, version, machine)
    _uname_cache = uname_result(*map(_unknown_as_blank, vals))
    return _uname_cache