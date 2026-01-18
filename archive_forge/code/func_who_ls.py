import gc
import re
import sys
from IPython.core import page
from IPython.core.error import StdinNotImplementedError, UsageError
from IPython.core.magic import Magics, magics_class, line_magic
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils.encoding import DEFAULT_ENCODING
from IPython.utils.openpy import read_py_file
from IPython.utils.path import get_py_filename
@skip_doctest
@line_magic
def who_ls(self, parameter_s=''):
    """Return a sorted list of all interactive variables.

        If arguments are given, only variables of types matching these
        arguments are returned.

        Examples
        --------
        Define two variables and list them with who_ls::

          In [1]: alpha = 123

          In [2]: beta = 'test'

          In [3]: %who_ls
          Out[3]: ['alpha', 'beta']

          In [4]: %who_ls int
          Out[4]: ['alpha']

          In [5]: %who_ls str
          Out[5]: ['beta']
        """
    user_ns = self.shell.user_ns
    user_ns_hidden = self.shell.user_ns_hidden
    nonmatching = object()
    out = [i for i in user_ns if not i.startswith('_') and user_ns[i] is not user_ns_hidden.get(i, nonmatching)]
    typelist = parameter_s.split()
    if typelist:
        typeset = set(typelist)
        out = [i for i in out if type(user_ns[i]).__name__ in typeset]
    out.sort()
    return out