from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import pkgutil
import textwrap
from googlecloudsdk import api_lib
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.resource import resource_transform
import six
from six.moves import range  # pylint: disable=redefined-builtin
def _ParseTransformDocString(func):
    """Parses the doc string for func.

  Args:
    func: The doc string will be parsed from this function.

  Returns:
    A (description, prototype, args) tuple:
      description - The function description.
      prototype - The function prototype string.
      args - A list of (name, description) tuples, one tuple for each arg.

  Example transform function docstring parsed by this method:
    '''Transform description. Example sections optional.

    These lines are skipped.
    Another skipped line.

    Args:
      r: The resource arg is always sepcified but omitted from the docs.
      arg-2-name[=default-2]: The description for arg-2-name.
      arg-N-name[=default-N]: The description for arg-N-name.
      kwargs: Omitted from the description.

    Example:
      One or more example lines for the 'For example:' section.
    '''
  """
    hidden_args = ('kwargs', 'projection', 'r')
    if not func.__doc__:
        return ('', '', '', '')
    description, _, doc = func.__doc__.partition('\n')
    collect = _DOC_MAIN
    arg = None
    descriptions = [description]
    example = []
    args = []
    arg_description = []
    paragraph = False
    for line in textwrap.dedent(doc).split('\n'):
        if not line:
            paragraph = True
        elif line == 'Args:':
            collect = _DOC_ARGS
            paragraph = False
        elif line == 'Example:':
            collect = _DOC_EXAMPLE
            paragraph = False
        elif collect == _DOC_SKIP:
            continue
        elif collect == _DOC_MAIN:
            paragraph = _AppendLine(descriptions, line, paragraph)
        elif collect == _DOC_ARGS and line.startswith('    '):
            paragraph = _AppendLine(arg_description, line, paragraph)
        elif collect == _DOC_EXAMPLE and line.startswith('  '):
            paragraph = _AppendLine(example, line[2:], paragraph)
        else:
            if arg:
                arg = _StripUnusedNotation(arg)
            if arg and arg not in hidden_args:
                args.append((arg, ' '.join(arg_description)))
            if not line.startswith(' ') and line.endswith(':'):
                collect = _DOC_SKIP
                continue
            arg, _, text = line.partition(':')
            arg = arg.strip()
            arg = arg.lstrip('*')
            arg_description = [text.strip()]
    import inspect
    argspec = inspect.getfullargspec(func)
    default_index_start = len(argspec.args) - len(argspec.defaults or [])
    formals = []
    for formal_index, formal in enumerate(argspec.args):
        if formal:
            formal = _StripUnusedNotation(formal)
        if formal in hidden_args:
            continue
        default_index = formal_index - default_index_start
        default = argspec.defaults[default_index] if default_index >= 0 else None
        if default is not None:
            default_display = repr(default).replace("'", '"')
            if default_display.startswith('u"'):
                default_display = default_display[1:]
            if default_display == 'False':
                default_display = 'false'
            elif default_display == 'True':
                default_display = 'true'
            formals.append('{formal}={default_display}'.format(formal=formal, default_display=default_display))
        else:
            formals.append(formal)
    if argspec.varargs:
        formals.append(argspec.varargs)
    prototype = '({formals})'.format(formals=', '.join(formals))
    return (''.join(descriptions), prototype, args, example)