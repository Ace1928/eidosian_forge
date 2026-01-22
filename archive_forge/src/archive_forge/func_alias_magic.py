from logging import error
import io
import os
from pprint import pformat
import sys
from warnings import warn
from traitlets.utils.importstring import import_item
from IPython.core import magic_arguments, page
from IPython.core.error import UsageError
from IPython.core.magic import Magics, magics_class, line_magic, magic_escapes
from IPython.utils.text import format_screen, dedent, indent
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils.ipstruct import Struct
@skip_doctest
@magic_arguments.magic_arguments()
@magic_arguments.argument('-l', '--line', action='store_true', help='Create a line magic alias.')
@magic_arguments.argument('-c', '--cell', action='store_true', help='Create a cell magic alias.')
@magic_arguments.argument('name', help='Name of the magic to be created.')
@magic_arguments.argument('target', help='Name of the existing line or cell magic.')
@magic_arguments.argument('-p', '--params', default=None, help='Parameters passed to the magic function.')
@line_magic
def alias_magic(self, line=''):
    """Create an alias for an existing line or cell magic.

        Examples
        --------
        ::

          In [1]: %alias_magic t timeit
          Created `%t` as an alias for `%timeit`.
          Created `%%t` as an alias for `%%timeit`.

          In [2]: %t -n1 pass
          107 ns ± 43.6 ns per loop (mean ± std. dev. of 7 runs, 1 loop each)

          In [3]: %%t -n1
             ...: pass
             ...:
          107 ns ± 58.3 ns per loop (mean ± std. dev. of 7 runs, 1 loop each)

          In [4]: %alias_magic --cell whereami pwd
          UsageError: Cell magic function `%%pwd` not found.
          In [5]: %alias_magic --line whereami pwd
          Created `%whereami` as an alias for `%pwd`.

          In [6]: %whereami
          Out[6]: '/home/testuser'

          In [7]: %alias_magic h history -p "-l 30" --line
          Created `%h` as an alias for `%history -l 30`.
        """
    args = magic_arguments.parse_argstring(self.alias_magic, line)
    shell = self.shell
    mman = self.shell.magics_manager
    escs = ''.join(magic_escapes.values())
    target = args.target.lstrip(escs)
    name = args.name.lstrip(escs)
    params = args.params
    if params and (params.startswith('"') and params.endswith('"') or (params.startswith("'") and params.endswith("'"))):
        params = params[1:-1]
    m_line = shell.find_magic(target, 'line')
    m_cell = shell.find_magic(target, 'cell')
    if args.line and m_line is None:
        raise UsageError('Line magic function `%s%s` not found.' % (magic_escapes['line'], target))
    if args.cell and m_cell is None:
        raise UsageError('Cell magic function `%s%s` not found.' % (magic_escapes['cell'], target))
    if not args.line and (not args.cell):
        if not m_line and (not m_cell):
            raise UsageError('No line or cell magic with name `%s` found.' % target)
        args.line = bool(m_line)
        args.cell = bool(m_cell)
    params_str = '' if params is None else ' ' + params
    if args.line:
        mman.register_alias(name, target, 'line', params)
        print('Created `%s%s` as an alias for `%s%s%s`.' % (magic_escapes['line'], name, magic_escapes['line'], target, params_str))
    if args.cell:
        mman.register_alias(name, target, 'cell', params)
        print('Created `%s%s` as an alias for `%s%s%s`.' % (magic_escapes['cell'], name, magic_escapes['cell'], target, params_str))