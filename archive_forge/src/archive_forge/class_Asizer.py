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
class Asizer(object):
    """Sizer state and options to accumulate sizes."""
    _above_ = 1024
    _align_ = 8
    _clip_ = 80
    _code_ = False
    _cutoff_ = 0
    _derive_ = False
    _detail_ = 0
    _frames_ = False
    _infer_ = False
    _limit_ = 100
    _stats_ = 0
    _depth = 0
    _excl_d = None
    _ign_d = _kind_ignored
    _incl = _NN
    _mask = 7
    _missed = 0
    _profile = False
    _profs = None
    _ranked = 0
    _ranks = []
    _seen = None
    _stream = None
    _total = 0

    def __init__(self, **opts):
        """New **Asizer** accumulator.

        See this module documentation for more details.
        See method **reset** for all available options and defaults.
        """
        self._excl_d = {}
        self.reset(**opts)

    def _c100(self, stats):
        """Cutoff as percentage (for backward compatibility)"""
        s = int(stats)
        c = int((stats - s) * 100.0 + 0.5) or self.cutoff
        return (s, c)

    def _clear(self):
        """Clear state."""
        self._depth = 0
        self._incl = _NN
        self._missed = 0
        self._profile = False
        self._profs = {}
        self._ranked = 0
        self._ranks = []
        self._seen = _Seen()
        self._total = 0
        for k in _keys(self._excl_d):
            self._excl_d[k] = 0
        m = sys.modules[__name__]
        self.exclude_objs(self, self._excl_d, self._profs, self._ranks, self._seen, m, m.__dict__, m.__doc__, _typedefs)

    def _nameof(self, obj):
        """Return the object's name."""
        return _nameof(obj, _NN) or self._repr(obj)

    def _prepr(self, obj):
        """Like **prepr()**."""
        return _prepr(obj, clip=self._clip_)

    def _printf(self, fmt, *args, **print3options):
        """Print to sys.stdout or the configured stream if any is
        specified and if the file keyword argument is not already
        set in the **print3options** for this specific call.
        """
        if self._stream and (not print3options.get('file', None)):
            if args:
                fmt = fmt % args
            _printf(fmt, file=self._stream, **print3options)
        else:
            _printf(fmt, *args, **print3options)

    def _prof(self, key):
        """Get _Prof object."""
        p = self._profs.get(key, None)
        if not p:
            self._profs[key] = p = _Prof()
            self.exclude_objs(p)
        return p

    def _rank(self, key, obj, size, deep, pid):
        """Rank 100 largest objects by size."""
        rs = self._ranks
        i, j = (0, len(rs))
        while i < j:
            m = (i + j) // 2
            if size < rs[m].size:
                i = m + 1
            else:
                j = m
        if i < 100:
            r = _Rank(key, obj, size, deep, pid)
            rs.insert(i, r)
            self.exclude_objs(r)
            while len(rs) > 100:
                rs.pop()
        self._ranked += 1

    def _repr(self, obj):
        """Like ``repr()``."""
        return _repr(obj, clip=self._clip_)

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

    def _sizes(self, objs, sized=None):
        """Return the size or an **Asized** instance for each
        given object plus the total size.  The total includes
        the size of duplicates only once.
        """
        self.exclude_refs(*objs)
        s, t = ({}, [])
        self.exclude_objs(s, t)
        for o in objs:
            i = id(o)
            if i in s:
                self._seen.again(i)
            else:
                s[i] = self._sizer(o, 0, 0, sized)
            t.append(s[i])
        return tuple(t)

    @property
    def above(self):
        """Get the large object size threshold (int)."""
        return self._above_

    @property
    def align(self):
        """Get the size alignment (int)."""
        return self._align_

    def asized(self, *objs, **opts):
        """Size each object and return an **Asized** instance with
        size information and referents up to the given detail
        level (and with modified options, see method **set**).

        If only one object is given, the return value is the
        **Asized** instance for that object.  The **Asized** size
        of duplicate and ignored objects will be zero.
        """
        if opts:
            self.set(**opts)
        t = self._sizes(objs, Asized)
        return t[0] if len(t) == 1 else t

    def asizeof(self, *objs, **opts):
        """Return the combined size of the given objects
        (with modified options, see method **set**).
        """
        if opts:
            self.set(**opts)
        self.exclude_refs(*objs)
        return sum((self._sizer(o, 0, 0, None) for o in objs))

    def asizesof(self, *objs, **opts):
        """Return the individual sizes of the given objects
        (with modified options, see method  **set**).

        The size of duplicate and ignored objects will be zero.
        """
        if opts:
            self.set(**opts)
        return self._sizes(objs, None)

    @property
    def clip(self):
        """Get the clipped string length (int)."""
        return self._clip_

    @property
    def code(self):
        """Size (byte) code (bool)."""
        return self._code_

    @property
    def cutoff(self):
        """Stats cutoff (int)."""
        return self._cutoff_

    @property
    def derive(self):
        """Derive types (bool)."""
        return self._derive_

    @property
    def detail(self):
        """Get the detail level for **Asized** refs (int)."""
        return self._detail_

    @property
    def duplicate(self):
        """Get the number of duplicate objects seen so far (int)."""
        return sum((1 for v in _values(self._seen) if v > 1))

    def exclude_objs(self, *objs):
        """Exclude the specified objects from sizing, profiling and ranking."""
        for o in objs:
            self._seen.setdefault(id(o), -1)

    def exclude_refs(self, *objs):
        """Exclude any references to the specified objects from sizing.

        While any references to the given objects are excluded, the
        objects will be sized if specified as positional arguments
        in subsequent calls to methods **asizeof** and **asizesof**.
        """
        for o in objs:
            self._seen.setdefault(id(o), 0)

    def exclude_types(self, *objs):
        """Exclude the specified object instances and types from sizing.

        All instances and types of the given objects are excluded,
        even objects specified as positional arguments in subsequent
        calls to methods **asizeof** and **asizesof**.
        """
        for o in objs:
            for t in _key2tuple(o):
                if t and t not in self._excl_d:
                    self._excl_d[t] = 0

    @property
    def excluded(self):
        """Get the types being excluded (tuple)."""
        return tuple(_keys(self._excl_d))

    @property
    def frames(self):
        """Ignore stack frames (bool)."""
        return self._frames_

    @property
    def ignored(self):
        """Ignore certain types (bool)."""
        return True if self._ign_d else False

    @property
    def infer(self):
        """Infer types (bool)."""
        return self._infer_

    @property
    def limit(self):
        """Get the recursion limit (int)."""
        return self._limit_

    @property
    def missed(self):
        """Get the number of objects missed due to errors (int)."""
        return self._missed

    def print_largest(self, w=0, cutoff=0, **print3options):
        """Print the largest objects.

        The available options and defaults are:

         *w=0*           -- indentation for each line

         *cutoff=100*    -- number of largest objects to print

         *print3options* -- some keyword arguments, like Python 3+ print
        """
        c = int(cutoff) if cutoff else self._cutoff_
        n = min(len(self._ranks), max(c, 0))
        s = self._above_
        if n > 0 and s > 0:
            self._printf('%s%*d largest object%s (of %d over %d bytes%s)', linesep, w, n, _plural(n), self._ranked, s, _SI(s), **print3options)
            id2x = dict(((r.id, i) for i, r in enumerate(self._ranks)))
            for r in self._ranks[:n]:
                s, t = (r.size, r.format(self._clip_, id2x))
                self._printf('%*d bytes%s: %s', w, s, _SI(s), t, **print3options)

    def print_profiles(self, w=0, cutoff=0, **print3options):
        """Print the profiles above *cutoff* percentage.

        The available options and defaults are:

             *w=0*           -- indentation for each line

             *cutoff=0*      -- minimum percentage printed

             *print3options* -- some keyword arguments, like Python 3+ print
        """
        t = [(v, k) for k, v in _items(self._profs) if v.total > 0 or v.number > 1]
        if len(self._profs) - len(t) < 9:
            t = [(v, k) for k, v in _items(self._profs)]
        if t:
            s = _NN
            if self._total:
                s = ' (% of grand total)'
                c = int(cutoff) if cutoff else self._cutoff_
                C = int(c * 0.01 * self._total)
            else:
                C = c = 0
            self._printf('%s%*d profile%s:  total%s, average, and largest flat size%s:  largest object', linesep, w, len(t), _plural(len(t)), s, self._incl, **print3options)
            r = len(t)
            t = [(v, self._prepr(k)) for v, k in t]
            for v, k in sorted(t, reverse=True):
                s = 'object%(plural)s:  %(total)s, %(avg)s, %(high)s:  %(obj)s%(lengstr)s' % v.format(self._clip_, self._total)
                self._printf('%*d %s %s', w, v.number, k, s, **print3options)
                r -= 1
                if r > 1 and v.total < C:
                    self._printf('%+*d profiles below cutoff (%.0f%%)', w, r, c)
                    break
            z = len(self._profs) - len(t)
            if z > 0:
                self._printf('%+*d %r object%s', w, z, 'zero', _plural(z), **print3options)

    def print_stats(self, objs=(), opts={}, sized=(), sizes=(), stats=3, **print3options):
        """Prints the statistics.

        The available options and defaults are:

             *w=0*           -- indentation for each line

             *objs=()*       -- optional, list of objects

             *opts={}*       -- optional, dict of options used

             *sized=()*      -- optional, tuple of **Asized** instances returned

             *sizes=()*      -- optional, tuple of sizes returned

             *stats=3*       -- print stats, see function **asizeof**

             *print3options* -- some keyword arguments, like Python 3+ print
        """
        s = min(opts.get('stats', stats) or 0, self.stats)
        if s > 0:
            w = len(str(self.missed + self.seen + self.total)) + 1
            t = c = _NN
            o = _kwdstr(**opts)
            if o and objs:
                c = ', '
            if sized and objs:
                n = len(objs)
                if n > 1:
                    self._printf('%sasized(...%s%s) ...', linesep, c, o, **print3options)
                    for i in range(n):
                        self._printf('%*d: %s', w - 1, i, sized[i], **print3options)
                else:
                    self._printf('%sasized(%s): %s', linesep, o, sized, **print3options)
            elif sizes and objs:
                self._printf('%sasizesof(...%s%s) ...', linesep, c, o, **print3options)
                for z, o in zip(sizes, objs):
                    self._printf('%*d bytes%s%s:  %s', w, z, _SI(z), self._incl, self._repr(o), **print3options)
            else:
                if objs:
                    t = self._repr(objs)
                self._printf('%sasizeof(%s%s%s) ...', linesep, t, c, o, **print3options)
            self.print_summary(w=w, objs=objs, **print3options)
            s, c = self._c100(s)
            self.print_largest(w=w, cutoff=c if s < 2 else 10, **print3options)
            if s > 1:
                self.print_profiles(w=w, cutoff=c, **print3options)
                if s > 2:
                    self.print_typedefs(w=w, **print3options)

    def print_summary(self, w=0, objs=(), **print3options):
        """Print the summary statistics.

        The available options and defaults are:

             *w=0*           -- indentation for each line

             *objs=()*       -- optional, list of objects

             *print3options* -- some keyword arguments, like Python 3+ print
        """
        self._printf('%*d bytes%s%s', w, self._total, _SI(self._total), self._incl, **print3options)
        if self._mask:
            self._printf('%*d byte aligned', w, self._mask + 1, **print3options)
        self._printf('%*d byte sizeof(void*)', w, _sizeof_Cvoidp, **print3options)
        n = len(objs or ())
        self._printf('%*d object%s %s', w, n, _plural(n), 'given', **print3options)
        n = self.sized
        self._printf('%*d object%s %s', w, n, _plural(n), 'sized', **print3options)
        if self._excl_d:
            n = sum(_values(self._excl_d))
            self._printf('%*d object%s %s', w, n, _plural(n), 'excluded', **print3options)
        n = self.seen
        self._printf('%*d object%s %s', w, n, _plural(n), 'seen', **print3options)
        n = self.ranked
        if n > 0:
            self._printf('%*d object%s %s', w, n, _plural(n), 'ranked', **print3options)
        n = self.missed
        self._printf('%*d object%s %s', w, n, _plural(n), 'missed', **print3options)
        n = self.duplicate
        self._printf('%*d duplicate%s', w, n, _plural(n), **print3options)
        if self._depth > 0:
            self._printf('%*d deepest recursion', w, self._depth, **print3options)

    def print_typedefs(self, w=0, **print3options):
        """Print the types and dict tables.

        The available options and defaults are:

             *w=0*           -- indentation for each line

             *print3options* -- some keyword arguments, like Python 3+ print
        """
        for k in _all_kinds:
            t = [(self._prepr(a), v) for a, v in _items(_typedefs) if v.kind == k and (v.both or self._code_)]
            if t:
                self._printf('%s%*d %s type%s:  basicsize, itemsize, _len_(), _refs()', linesep, w, len(t), k, _plural(len(t)), **print3options)
                for a, v in sorted(t):
                    self._printf('%*s %s:  %s', w, _NN, a, v, **print3options)
        t = sum((len(v) for v in _values(_dict_types)))
        if t:
            self._printf('%s%*d dict/-like classes:', linesep, w, t, **print3options)
            for m, v in _items(_dict_types):
                self._printf('%*s %s:  %s', w, _NN, m, self._prepr(v), **print3options)

    @property
    def ranked(self):
        """Get the number objects ranked by size so far (int)."""
        return self._ranked

    def reset(self, above=1024, align=8, clip=80, code=False, cutoff=10, derive=False, detail=0, frames=False, ignored=True, infer=False, limit=100, stats=0, stream=None, **extra):
        """Reset sizing options, state, etc. to defaults.

        The available options and default values are:

             *above=0*      -- threshold for largest objects stats

             *align=8*      -- size alignment

             *code=False*   -- incl. (byte)code size

             *cutoff=10*    -- limit large objects or profiles stats

             *derive=False* -- derive from super type

             *detail=0*     -- **Asized** refs level

             *frames=False* -- ignore frame objects

             *ignored=True* -- ignore certain types

             *infer=False*  -- try to infer types

             *limit=100*    -- recursion limit

             *stats=0*      -- print statistics, see function **asizeof**

             *stream=None*  -- output stream for printing

        See function **asizeof** for a description of the options.
        """
        if extra:
            raise _OptionError(self.reset, Error=KeyError, **extra)
        self._above_ = above
        self._align_ = align
        self._clip_ = clip
        self._code_ = code
        self._cutoff_ = cutoff
        self._derive_ = derive
        self._detail_ = detail
        self._frames_ = frames
        self._infer_ = infer
        self._limit_ = limit
        self._stats_ = stats
        self._stream = stream
        if ignored:
            self._ign_d = _kind_ignored
        else:
            self._ign_d = None
        self._clear()
        self.set(align=align, code=code, cutoff=cutoff, stats=stats)

    @property
    def seen(self):
        """Get the number objects seen so far (int)."""
        return sum((v for v in _values(self._seen) if v > 0))

    def set(self, above=None, align=None, code=None, cutoff=None, frames=None, detail=None, limit=None, stats=None):
        """Set some sizing options.  See also **reset**.

        The available options are:

             *above*  -- threshold for largest objects stats

             *align*  -- size alignment

             *code*   -- incl. (byte)code size

             *cutoff* -- limit large objects or profiles stats

             *detail* -- **Asized** refs level

             *frames* -- size or ignore frame objects

             *limit*  -- recursion limit

             *stats*  -- print statistics, see function **asizeof**

        Any options not set remain unchanged from the previous setting.
        """
        if above is not None:
            self._above_ = int(above)
        if align is not None:
            if align > 1:
                m = align - 1
                if m & align:
                    raise _OptionError(self.set, align=align)
            else:
                m = 0
            self._align_ = align
            self._mask = m
        if code is not None:
            self._code_ = code
            if code:
                self._incl = ' (incl. code)'
        if detail is not None:
            self._detail_ = detail
        if frames is not None:
            self._frames_ = frames
        if limit is not None:
            self._limit_ = limit
        if stats is not None:
            if stats < 0:
                raise _OptionError(self.set, stats=stats)
            s, c = self._c100(stats)
            self._cutoff_ = int(cutoff) if cutoff else c
            self._stats_ = s
            self._profile = s > 1

    @property
    def sized(self):
        """Get the number objects sized so far (int)."""
        return sum((1 for v in _values(self._seen) if v > 0))

    @property
    def stats(self):
        """Get the stats and cutoff setting (float)."""
        return self._stats_

    @property
    def total(self):
        """Get the total size (in bytes) accumulated so far."""
        return self._total