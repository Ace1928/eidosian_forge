import collections
import platform
import sys
def _implementation_tuple():
    """Return the tuple of interpreter name and version.

    Returns a string that provides both the name and the version of the Python
    implementation currently running. For example, on CPython 2.7.5 it will
    return "CPython/2.7.5".

    This function works best on CPython and PyPy: in particular, it probably
    doesn't work for Jython or IronPython. Future investigation should be done
    to work out the correct shape of the code for those platforms.
    """
    implementation = platform.python_implementation()
    if implementation == 'CPython':
        implementation_version = platform.python_version()
    elif implementation == 'PyPy':
        implementation_version = '%s.%s.%s' % (sys.pypy_version_info.major, sys.pypy_version_info.minor, sys.pypy_version_info.micro)
        if sys.pypy_version_info.releaselevel != 'final':
            implementation_version = ''.join([implementation_version, sys.pypy_version_info.releaselevel])
    elif implementation == 'Jython':
        implementation_version = platform.python_version()
    elif implementation == 'IronPython':
        implementation_version = platform.python_version()
    else:
        implementation_version = 'Unknown'
    return (implementation, implementation_version)