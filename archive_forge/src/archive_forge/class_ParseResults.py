import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
class ParseResults(object):
    """Structured parse results, to provide multiple means of access to the parsed data:
       - as a list (C{len(results)})
       - by list index (C{results[0], results[1]}, etc.)
       - by attribute (C{results.<resultsName>})
       """

    def __new__(cls, toklist, name=None, asList=True, modal=True):
        if isinstance(toklist, cls):
            return toklist
        retobj = object.__new__(cls)
        retobj.__doinit = True
        return retobj

    def __init__(self, toklist, name=None, asList=True, modal=True, isinstance=isinstance):
        if self.__doinit:
            self.__doinit = False
            self.__name = None
            self.__parent = None
            self.__accumNames = {}
            if isinstance(toklist, list):
                self.__toklist = toklist[:]
            else:
                self.__toklist = [toklist]
            self.__tokdict = dict()
        if name is not None and name:
            if not modal:
                self.__accumNames[name] = 0
            if isinstance(name, int):
                name = _ustr(name)
            self.__name = name
            if not toklist in (None, '', []):
                if isinstance(toklist, basestring):
                    toklist = [toklist]
                if asList:
                    if isinstance(toklist, ParseResults):
                        self[name] = _ParseResultsWithOffset(toklist.copy(), 0)
                    else:
                        self[name] = _ParseResultsWithOffset(ParseResults(toklist[0]), 0)
                    self[name].__name = name
                else:
                    try:
                        self[name] = toklist[0]
                    except (KeyError, TypeError, IndexError):
                        self[name] = toklist

    def __getitem__(self, i):
        if isinstance(i, (int, slice)):
            return self.__toklist[i]
        elif i not in self.__accumNames:
            return self.__tokdict[i][-1][0]
        else:
            return ParseResults([v[0] for v in self.__tokdict[i]])

    def __setitem__(self, k, v, isinstance=isinstance):
        if isinstance(v, _ParseResultsWithOffset):
            self.__tokdict[k] = self.__tokdict.get(k, list()) + [v]
            sub = v[0]
        elif isinstance(k, int):
            self.__toklist[k] = v
            sub = v
        else:
            self.__tokdict[k] = self.__tokdict.get(k, list()) + [_ParseResultsWithOffset(v, 0)]
            sub = v
        if isinstance(sub, ParseResults):
            sub.__parent = wkref(self)

    def __delitem__(self, i):
        if isinstance(i, (int, slice)):
            mylen = len(self.__toklist)
            del self.__toklist[i]
            if isinstance(i, int):
                if i < 0:
                    i += mylen
                i = slice(i, i + 1)
            removed = list(range(*i.indices(mylen)))
            removed.reverse()
            for name in self.__tokdict:
                occurrences = self.__tokdict[name]
                for j in removed:
                    for k, (value, position) in enumerate(occurrences):
                        occurrences[k] = _ParseResultsWithOffset(value, position - (position > j))
        else:
            del self.__tokdict[i]

    def __contains__(self, k):
        return k in self.__tokdict

    def __len__(self):
        return len(self.__toklist)

    def __bool__(self):
        return len(self.__toklist) > 0
    __nonzero__ = __bool__

    def __iter__(self):
        return iter(self.__toklist)

    def __reversed__(self):
        return iter(self.__toklist[::-1])

    def keys(self):
        """Returns all named result keys."""
        return self.__tokdict.keys()

    def pop(self, index=-1):
        """Removes and returns item at specified index (default=last).
           Will work with either numeric indices or dict-key indicies."""
        ret = self[index]
        del self[index]
        return ret

    def get(self, key, defaultValue=None):
        """Returns named result matching the given key, or if there is no
           such name, then returns the given C{defaultValue} or C{None} if no
           C{defaultValue} is specified."""
        if key in self:
            return self[key]
        else:
            return defaultValue

    def insert(self, index, insStr):
        """Inserts new element at location index in the list of parsed tokens."""
        self.__toklist.insert(index, insStr)
        for name in self.__tokdict:
            occurrences = self.__tokdict[name]
            for k, (value, position) in enumerate(occurrences):
                occurrences[k] = _ParseResultsWithOffset(value, position + (position > index))

    def items(self):
        """Returns all named result keys and values as a list of tuples."""
        return [(k, self[k]) for k in self.__tokdict]

    def values(self):
        """Returns all named result values."""
        return [v[-1][0] for v in self.__tokdict.values()]

    def __getattr__(self, name):
        if True:
            if name in self.__tokdict:
                if name not in self.__accumNames:
                    return self.__tokdict[name][-1][0]
                else:
                    return ParseResults([v[0] for v in self.__tokdict[name]])
            else:
                return ''
        return None

    def __add__(self, other):
        ret = self.copy()
        ret += other
        return ret

    def __iadd__(self, other):
        if other.__tokdict:
            offset = len(self.__toklist)
            addoffset = lambda a: a < 0 and offset or a + offset
            otheritems = other.__tokdict.items()
            otherdictitems = [(k, _ParseResultsWithOffset(v[0], addoffset(v[1]))) for k, vlist in otheritems for v in vlist]
            for k, v in otherdictitems:
                self[k] = v
                if isinstance(v[0], ParseResults):
                    v[0].__parent = wkref(self)
        self.__toklist += other.__toklist
        self.__accumNames.update(other.__accumNames)
        return self

    def __radd__(self, other):
        if isinstance(other, int) and other == 0:
            return self.copy()

    def __repr__(self):
        return '(%s, %s)' % (repr(self.__toklist), repr(self.__tokdict))

    def __str__(self):
        out = []
        for i in self.__toklist:
            if isinstance(i, ParseResults):
                out.append(_ustr(i))
            else:
                out.append(repr(i))
        return '[' + ', '.join(out) + ']'

    def _asStringList(self, sep=''):
        out = []
        for item in self.__toklist:
            if out and sep:
                out.append(sep)
            if isinstance(item, ParseResults):
                out += item._asStringList()
            else:
                out.append(_ustr(item))
        return out

    def asList(self):
        """Returns the parse results as a nested list of matching tokens, all converted to strings."""
        out = []
        for res in self.__toklist:
            if isinstance(res, ParseResults):
                out.append(res.asList())
            else:
                out.append(res)
        return out

    def asDict(self):
        """Returns the named parse results as dictionary."""
        return dict(self.items())

    def copy(self):
        """Returns a new copy of a C{ParseResults} object."""
        ret = ParseResults(self.__toklist)
        ret.__tokdict = self.__tokdict.copy()
        ret.__parent = self.__parent
        ret.__accumNames.update(self.__accumNames)
        ret.__name = self.__name
        return ret

    def asXML(self, doctag=None, namedItemsOnly=False, indent='', formatted=True):
        """Returns the parse results as XML. Tags are created for tokens and lists that have defined results names."""
        nl = '\n'
        out = []
        namedItems = dict(((v[1], k) for k, vlist in self.__tokdict.items() for v in vlist))
        nextLevelIndent = indent + '  '
        if not formatted:
            indent = ''
            nextLevelIndent = ''
            nl = ''
        selfTag = None
        if doctag is not None:
            selfTag = doctag
        elif self.__name:
            selfTag = self.__name
        if not selfTag:
            if namedItemsOnly:
                return ''
            else:
                selfTag = 'ITEM'
        out += [nl, indent, '<', selfTag, '>']
        worklist = self.__toklist
        for i, res in enumerate(worklist):
            if isinstance(res, ParseResults):
                if i in namedItems:
                    out += [res.asXML(namedItems[i], namedItemsOnly and doctag is None, nextLevelIndent, formatted)]
                else:
                    out += [res.asXML(None, namedItemsOnly and doctag is None, nextLevelIndent, formatted)]
            else:
                resTag = None
                if i in namedItems:
                    resTag = namedItems[i]
                if not resTag:
                    if namedItemsOnly:
                        continue
                    else:
                        resTag = 'ITEM'
                xmlBodyText = _xml_escape(_ustr(res))
                out += [nl, nextLevelIndent, '<', resTag, '>', xmlBodyText, '</', resTag, '>']
        out += [nl, indent, '</', selfTag, '>']
        return ''.join(out)

    def __lookup(self, sub):
        for k, vlist in self.__tokdict.items():
            for v, loc in vlist:
                if sub is v:
                    return k
        return None

    def getName(self):
        """Returns the results name for this token expression."""
        if self.__name:
            return self.__name
        elif self.__parent:
            par = self.__parent()
            if par:
                return par.__lookup(self)
            else:
                return None
        elif len(self) == 1 and len(self.__tokdict) == 1 and (self.__tokdict.values()[0][0][1] in (0, -1)):
            return self.__tokdict.keys()[0]
        else:
            return None

    def dump(self, indent='', depth=0):
        """Diagnostic method for listing out the contents of a C{ParseResults}.
           Accepts an optional C{indent} argument so that this string can be embedded
           in a nested display of other data."""
        out = []
        out.append(indent + _ustr(self.asList()))
        keys = self.items()
        keys.sort()
        for k, v in keys:
            if out:
                out.append('\n')
            out.append('%s%s- %s: ' % (indent, '  ' * depth, k))
            if isinstance(v, ParseResults):
                if v.keys():
                    out.append(v.dump(indent, depth + 1))
                else:
                    out.append(_ustr(v))
            else:
                out.append(_ustr(v))
        return ''.join(out)

    def __getstate__(self):
        return (self.__toklist, (self.__tokdict.copy(), self.__parent is not None and self.__parent() or None, self.__accumNames, self.__name))

    def __setstate__(self, state):
        self.__toklist = state[0]
        self.__tokdict, par, inAccumNames, self.__name = state[1]
        self.__accumNames = {}
        self.__accumNames.update(inAccumNames)
        if par is not None:
            self.__parent = wkref(par)
        else:
            self.__parent = None

    def __dir__(self):
        return dir(super(ParseResults, self)) + list(self.keys())