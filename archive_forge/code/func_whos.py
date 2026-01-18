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
def whos(self, parameter_s=''):
    """Like %who, but gives some extra information about each variable.

        The same type filtering of %who can be applied here.

        For all variables, the type is printed. Additionally it prints:

          - For {},[],(): their length.

          - For numpy arrays, a summary with shape, number of
            elements, typecode and size in memory.

          - Everything else: a string representation, snipping their middle if
            too long.

        Examples
        --------
        Define two variables and list them with whos::

          In [1]: alpha = 123

          In [2]: beta = 'test'

          In [3]: %whos
          Variable   Type        Data/Info
          --------------------------------
          alpha      int         123
          beta       str         test
        """
    varnames = self.who_ls(parameter_s)
    if not varnames:
        if parameter_s:
            print('No variables match your requested type.')
        else:
            print('Interactive namespace is empty.')
        return
    seq_types = ['dict', 'list', 'tuple']
    ndarray_type = None
    if 'numpy' in sys.modules:
        try:
            from numpy import ndarray
        except ImportError:
            pass
        else:
            ndarray_type = ndarray.__name__
    abbrevs = {'IPython.core.macro.Macro': 'Macro'}

    def type_name(v):
        tn = type(v).__name__
        return abbrevs.get(tn, tn)
    varlist = [self.shell.user_ns[n] for n in varnames]
    typelist = []
    for vv in varlist:
        tt = type_name(vv)
        if tt == 'instance':
            typelist.append(abbrevs.get(str(vv.__class__), str(vv.__class__)))
        else:
            typelist.append(tt)
    varlabel = 'Variable'
    typelabel = 'Type'
    datalabel = 'Data/Info'
    colsep = 3
    vformat = '{0:<{varwidth}}{1:<{typewidth}}'
    aformat = '%s: %s elems, type `%s`, %s bytes'
    varwidth = max(max(map(len, varnames)), len(varlabel)) + colsep
    typewidth = max(max(map(len, typelist)), len(typelabel)) + colsep
    print(varlabel.ljust(varwidth) + typelabel.ljust(typewidth) + ' ' + datalabel + '\n' + '-' * (varwidth + typewidth + len(datalabel) + 1))
    kb = 1024
    Mb = 1048576
    for vname, var, vtype in zip(varnames, varlist, typelist):
        print(vformat.format(vname, vtype, varwidth=varwidth, typewidth=typewidth), end=' ')
        if vtype in seq_types:
            print('n=' + str(len(var)))
        elif vtype == ndarray_type:
            vshape = str(var.shape).replace(',', '').replace(' ', 'x')[1:-1]
            if vtype == ndarray_type:
                vsize = var.size
                vbytes = vsize * var.itemsize
                vdtype = var.dtype
            if vbytes < 100000:
                print(aformat % (vshape, vsize, vdtype, vbytes))
            else:
                print(aformat % (vshape, vsize, vdtype, vbytes), end=' ')
                if vbytes < Mb:
                    print('(%s kb)' % (vbytes / kb,))
                else:
                    print('(%s Mb)' % (vbytes / Mb,))
        else:
            try:
                vstr = str(var)
            except UnicodeEncodeError:
                vstr = var.encode(DEFAULT_ENCODING, 'backslashreplace')
            except:
                vstr = '<object with id %d (str() failed)>' % id(var)
            vstr = vstr.replace('\n', '\\n')
            if len(vstr) < 50:
                print(vstr)
            else:
                print(vstr[:25] + '<...>' + vstr[-25:])