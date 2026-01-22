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
def _sizer(self, obj, pid, deep, sized):
    """Size an object, recursively."""
    s, f, i = (0, 0, id(obj))
    if i not in self._seen:
        self._seen[i] = 1
    elif deep or self._seen[i]:
        self._seen.again(i)
        if sized:
            s = sized(s, f, name=self._nameof(obj))
            self.exclude_objs(s)
        return s
    else:
        self._seen.again(i)
    try:
        k, rs = (_objkey(obj), [])
        if k in self._excl_d:
            self._excl_d[k] += 1
        else:
            v = _typedefs.get(k, None)
            if not v:
                _typedefs[k] = v = _typedef(obj, derive=self._derive_, frames=self._frames_, infer=self._infer_)
            if (v.both or self._code_) and v.kind is not self._ign_d:
                s = f = v.flat(obj, self._mask)
                if self._profile:
                    self._prof(k).update(obj, s)
                if v.refs and deep < self._limit_ and (not (deep and ismodule(obj))):
                    z, d = (self._sizer, deep + 1)
                    if sized and deep < self._detail_:
                        self.exclude_objs(rs)
                        for o in v.refs(obj, True):
                            if isinstance(o, _NamedRef):
                                r = z(o.ref, i, d, sized)
                                r.name = o.name
                            else:
                                r = z(o, i, d, sized)
                                r.name = self._nameof(o)
                            rs.append(r)
                            s += r.size
                    else:
                        for o in v.refs(obj, False):
                            s += z(o, i, d, None)
                    if self._depth < d:
                        self._depth = d
            if self._stats_ and s > self._above_ > 0:
                self._rank(k, obj, s, deep, pid)
    except RuntimeError:
        self._missed += 1
    if not deep:
        self._total += s
    if sized:
        s = sized(s, f, name=self._nameof(obj), refs=rs)
        self.exclude_objs(s)
    return s