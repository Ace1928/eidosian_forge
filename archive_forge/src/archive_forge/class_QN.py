import collections
import gast
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import parser
class QN(object):
    """Represents a qualified name."""

    def __init__(self, base, attr=None, subscript=None):
        if attr is not None and subscript is not None:
            raise ValueError('A QN can only be either an attr or a subscript, not both: attr={}, subscript={}.'.format(attr, subscript))
        self._has_attr = False
        self._has_subscript = False
        if attr is not None:
            if not isinstance(base, QN):
                raise ValueError('for attribute QNs, base must be a QN; got instead "%s"' % base)
            if not isinstance(attr, str):
                raise ValueError('attr may only be a string; got instead "%s"' % attr)
            self._parent = base
            self.qn = (base, attr)
            self._has_attr = True
        elif subscript is not None:
            if not isinstance(base, QN):
                raise ValueError('For subscript QNs, base must be a QN.')
            self._parent = base
            self.qn = (base, subscript)
            self._has_subscript = True
        else:
            if not isinstance(base, (str, Literal)):
                raise ValueError('for simple QNs, base must be a string or a Literal object; got instead "%s"' % type(base))
            assert '.' not in base and '[' not in base and (']' not in base)
            self._parent = None
            self.qn = (base,)

    def is_symbol(self):
        return isinstance(self.qn[0], str)

    def is_simple(self):
        return len(self.qn) <= 1

    def is_composite(self):
        return len(self.qn) > 1

    def has_subscript(self):
        return self._has_subscript

    def has_attr(self):
        return self._has_attr

    @property
    def attr(self):
        if not self._has_attr:
            raise ValueError('Cannot get attr of non-attribute "%s".' % self)
        return self.qn[1]

    @property
    def parent(self):
        if self._parent is None:
            raise ValueError('Cannot get parent of simple name "%s".' % self.qn[0])
        return self._parent

    @property
    def owner_set(self):
        """Returns all the symbols (simple or composite) that own this QN.

    In other words, if this symbol was modified, the symbols in the owner set
    may also be affected.

    Examples:
      'a.b[c.d]' has two owners, 'a' and 'a.b'
    """
        owners = set()
        if self.has_attr() or self.has_subscript():
            owners.add(self.parent)
            owners.update(self.parent.owner_set)
        return owners

    @property
    def support_set(self):
        """Returns the set of simple symbols that this QN relies on.

    This would be the smallest set of symbols necessary for the QN to
    statically resolve (assuming properties and index ranges are verified
    at runtime).

    Examples:
      'a.b' has only one support symbol, 'a'
      'a[i]' has two support symbols, 'a' and 'i'
    """
        roots = set()
        if self.has_attr():
            roots.update(self.parent.support_set)
        elif self.has_subscript():
            roots.update(self.parent.support_set)
            roots.update(self.qn[1].support_set)
        else:
            roots.add(self)
        return roots

    def __hash__(self):
        return hash(self.qn + (self._has_attr, self._has_subscript))

    def __eq__(self, other):
        return isinstance(other, QN) and self.qn == other.qn and (self.has_subscript() == other.has_subscript()) and (self.has_attr() == other.has_attr())

    def __lt__(self, other):
        return str(self) < str(other)

    def __gt__(self, other):
        return str(self) > str(other)

    def __str__(self):
        root = self.qn[0]
        if self.has_subscript():
            return '{}[{}]'.format(root, self.qn[1])
        if self.has_attr():
            return '.'.join(map(str, self.qn))
        else:
            return str(root)

    def __repr__(self):
        return str(self)

    def ssf(self):
        """Simple symbol form."""
        ssfs = [n.ssf() if isinstance(n, QN) else n for n in self.qn]
        ssf_string = ''
        for i in range(0, len(self.qn) - 1):
            if self.has_subscript():
                delimiter = '_sub_'
            else:
                delimiter = '_'
            ssf_string += ssfs[i] + delimiter
        return ssf_string + ssfs[-1]

    def ast(self):
        """AST representation."""
        if self.has_subscript():
            return gast.Subscript(value=self.parent.ast(), slice=self.qn[-1].ast(), ctx=CallerMustSetThis)
        if self.has_attr():
            return gast.Attribute(value=self.parent.ast(), attr=self.qn[-1], ctx=CallerMustSetThis)
        base = self.qn[0]
        if isinstance(base, str):
            return gast.Name(base, ctx=CallerMustSetThis, annotation=None, type_comment=None)
        elif isinstance(base, Literal):
            return gast.Constant(base.value, kind=None)
        else:
            assert False, 'the constructor should prevent types other than str and Literal'