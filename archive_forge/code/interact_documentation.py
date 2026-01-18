from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import inspect
Drops into an IPython REPL with variables available for use.

  Args:
    variables: A dict of variables to make available. Keys are variable names.
        Values are variable values.
    argv: The argv to use for starting ipython. Defaults to an empty list.
  