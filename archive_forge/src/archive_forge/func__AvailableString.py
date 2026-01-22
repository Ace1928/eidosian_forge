from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import inspect
def _AvailableString(variables, verbose=False):
    """Returns a string describing what objects are available in the Python REPL.

  Args:
    variables: A dict of the object to be available in the REPL.
    verbose: Whether to include 'hidden' members, those keys starting with _.
  Returns:
    A string fit for printing at the start of the REPL, indicating what objects
    are available for the user to use.
  """
    modules = []
    other = []
    for name, value in variables.items():
        if not verbose and name.startswith('_'):
            continue
        if '-' in name or '/' in name:
            continue
        if inspect.ismodule(value):
            modules.append(name)
        else:
            other.append(name)
    lists = [('Modules', modules), ('Objects', other)]
    liststrs = []
    for name, varlist in lists:
        if varlist:
            liststrs.append('{name}: {items}'.format(name=name, items=', '.join(sorted(varlist))))
    return 'Fire is starting a Python REPL with the following objects:\n{liststrs}\n'.format(liststrs='\n'.join(liststrs))