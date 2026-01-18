import sys
from sys import version_info
Return an exception if running in an unsupported version of Python.

  This function compares the running version of cPython and against the list
  of supported python version. If the running version is less than any of the
  supported versions, return a Tuple of (False, Str(error message)) for the
  caller to handle. Minor versions of Python greater than those listed in the
  supported versions are allowed.

  Args:
    None
  Returns:
    Tuple(Boolean, String)

    A Tuple containing a Boolean and a String. The boolean represents if the
    version is supported, and the String will either be empty, or contain an
    error message.
  