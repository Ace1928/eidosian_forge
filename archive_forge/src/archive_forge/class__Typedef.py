import sys
import types as Types
import warnings
import weakref as Weakref
from inspect import isbuiltin, isclass, iscode, isframe, isfunction, ismethod, ismodule
from math import log
from os import curdir, linesep
from struct import calcsize
from gc import get_objects as _getobjects
from gc import get_referents as _getreferents  # containers only?
from array import array as _array  # array type
class _Typedef(object):
    """Type definition class."""
    base = 0
    both = None
    item = 0
    kind = None
    leng = None
    refs = None
    type = None
    vari = None
    xtyp = None

    def __init__(self, **kwds):
        self.reset(**kwds)

    def __lt__(self, unused):
        return True

    def __repr__(self):
        return repr(self.args())

    def __str__(self):
        t = [str(self.base), str(self.item)]
        for f in (self.leng, self.refs):
            t.append(_nameof(f) or 'n/a')
        if not self.both:
            t.append('(code only)')
        return ', '.join(t)

    def args(self):
        """Return all attributes as arguments tuple."""
        return (self.base, self.item, self.leng, self.refs, self.both, self.kind, self.type, self.xtyp)

    def dup(self, other=None, **kwds):
        """Duplicate attributes of dict or other typedef."""
        t = other or _dict_typedef
        d = t.kwds()
        d.update(kwds)
        self.reset(**d)

    def flat(self, obj, mask=0):
        """Return the aligned flat size."""
        s = self.base
        if self.leng and self.item > 0:
            s += self.leng(obj) * self.item
        if not self.xtyp:
            s = _getsizeof(obj, s)
        if mask:
            s = s + mask & ~mask
        return s

    def format(self):
        """Return format dict."""
        a = _nameof(self.leng)
        return dict(leng=' (%s)' % (a,) if a else _NN, item='var' if self.vari else self.item, code=_NN if self.both else ' (code only)', base=self.base, kind=self.kind)

    def kwds(self):
        """Return all attributes as keywords dict."""
        return dict(base=self.base, both=self.both, item=self.item, kind=self.kind, leng=self.leng, refs=self.refs, type=self.type, vari=self.vari, xtyp=self.xtyp)

    def reset(self, base=0, item=0, leng=None, refs=None, both=True, kind=None, type=None, vari=_Not_vari, xtyp=False, **extra):
        """Reset all specified typedef attributes."""
        v = vari or _Not_vari
        if v != str(v):
            e = dict(vari=v)
        elif base < 0:
            e = dict(base=base)
        elif both not in (False, True):
            e = dict(both=both)
        elif item < 0:
            e = dict(item=item)
        elif kind not in _all_kinds:
            e = dict(kind=kind)
        elif leng not in _all_lens:
            e = dict(leng=leng)
        elif refs not in _all_refs:
            e = dict(refs=refs)
        elif xtyp not in (False, True):
            e = dict(xtyp=xtyp)
        elif extra:
            e = {}
        else:
            self.base = base
            self.both = both
            self.item = item
            self.kind = kind
            self.leng = leng
            self.refs = refs
            self.type = type
            self.vari = v
            self.xtyp = xtyp
            return
        e.update(extra)
        raise _OptionError(self.reset, **e)

    def save(self, t, base=0, heap=False):
        """Save this typedef plus its class typedef."""
        c, k = _key2tuple(t)
        if k and k not in _typedefs:
            _typedefs[k] = self
            if c and c not in _typedefs:
                b = _basicsize(type(t), base=base, heap=heap)
                k = _kind_ignored if _isignored(t) else self.kind
                _typedefs[c] = _Typedef(base=b, both=False, kind=k, type=t, refs=_type_refs)
        elif t not in _typedefs:
            if not _isbuiltin2(t):
                s = ' '.join((self.vari, _moduleof(t), _nameof(t)))
                s = '%r %s %s' % ((c, k), self.both, s.strip())
                raise KeyError('typedef %r bad: %s' % (self, s))
            _typedefs[t] = _Typedef(base=_basicsize(t, base=base), both=False, kind=_kind_ignored, type=t)

    def set(self, safe_len=False, **kwds):
        """Set one or more attributes."""
        if kwds:
            d = self.kwds()
            d.update(kwds)
            self.reset(**d)
        if safe_len and self.item:
            self.leng = _len