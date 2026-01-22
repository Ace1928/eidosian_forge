from contextlib import contextmanager
import inspect
from sympy.core.symbol import Str
from sympy.core.sympify import _sympify
from sympy.logic.boolalg import Boolean, false, true
from sympy.multipledispatch.dispatcher import Dispatcher, str_signature
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import is_sequence
from sympy.utilities.source import get_class
class PredicateMeta(type):

    def __new__(cls, clsname, bases, dct):
        if 'handler' not in dct:
            name = f'Ask{clsname.capitalize()}Handler'
            handler = Dispatcher(name, doc='Handler for key %s' % name)
            dct['handler'] = handler
        dct['_orig_doc'] = dct.get('__doc__', '')
        return super().__new__(cls, clsname, bases, dct)

    @property
    def __doc__(cls):
        handler = cls.handler
        doc = cls._orig_doc
        if cls is not Predicate and handler is not None:
            doc += 'Handler\n'
            doc += '    =======\n\n'
            docs = ['    Multiply dispatched method: %s' % handler.name]
            if handler.doc:
                for line in handler.doc.splitlines():
                    if not line:
                        continue
                    docs.append('    %s' % line)
            other = []
            for sig in handler.ordering[::-1]:
                func = handler.funcs[sig]
                if func.__doc__:
                    s = '    Inputs: <%s>' % str_signature(sig)
                    lines = []
                    for line in func.__doc__.splitlines():
                        lines.append('    %s' % line)
                    s += '\n'.join(lines)
                    docs.append(s)
                else:
                    other.append(str_signature(sig))
            if other:
                othersig = '    Other signatures:'
                for line in other:
                    othersig += '\n        * %s' % line
                docs.append(othersig)
            doc += '\n\n'.join(docs)
        return doc