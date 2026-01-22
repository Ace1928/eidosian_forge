import re
import sys
import types
import copy
import os
import inspect
class LexerReflect(object):

    def __init__(self, ldict, log=None, reflags=0):
        self.ldict = ldict
        self.error_func = None
        self.tokens = []
        self.reflags = reflags
        self.stateinfo = {'INITIAL': 'inclusive'}
        self.modules = set()
        self.error = False
        self.log = PlyLogger(sys.stderr) if log is None else log

    def get_all(self):
        self.get_tokens()
        self.get_literals()
        self.get_states()
        self.get_rules()

    def validate_all(self):
        self.validate_tokens()
        self.validate_literals()
        self.validate_rules()
        return self.error

    def get_tokens(self):
        tokens = self.ldict.get('tokens', None)
        if not tokens:
            self.log.error('No token list is defined')
            self.error = True
            return
        if not isinstance(tokens, (list, tuple)):
            self.log.error('tokens must be a list or tuple')
            self.error = True
            return
        if not tokens:
            self.log.error('tokens is empty')
            self.error = True
            return
        self.tokens = tokens

    def validate_tokens(self):
        terminals = {}
        for n in self.tokens:
            if not _is_identifier.match(n):
                self.log.error("Bad token name '%s'", n)
                self.error = True
            if n in terminals:
                self.log.warning("Token '%s' multiply defined", n)
            terminals[n] = 1

    def get_literals(self):
        self.literals = self.ldict.get('literals', '')
        if not self.literals:
            self.literals = ''

    def validate_literals(self):
        try:
            for c in self.literals:
                if not isinstance(c, StringTypes) or len(c) > 1:
                    self.log.error('Invalid literal %s. Must be a single character', repr(c))
                    self.error = True
        except TypeError:
            self.log.error('Invalid literals specification. literals must be a sequence of characters')
            self.error = True

    def get_states(self):
        self.states = self.ldict.get('states', None)
        if self.states:
            if not isinstance(self.states, (tuple, list)):
                self.log.error('states must be defined as a tuple or list')
                self.error = True
            else:
                for s in self.states:
                    if not isinstance(s, tuple) or len(s) != 2:
                        self.log.error("Invalid state specifier %s. Must be a tuple (statename,'exclusive|inclusive')", repr(s))
                        self.error = True
                        continue
                    name, statetype = s
                    if not isinstance(name, StringTypes):
                        self.log.error('State name %s must be a string', repr(name))
                        self.error = True
                        continue
                    if not (statetype == 'inclusive' or statetype == 'exclusive'):
                        self.log.error("State type for state %s must be 'inclusive' or 'exclusive'", name)
                        self.error = True
                        continue
                    if name in self.stateinfo:
                        self.log.error("State '%s' already defined", name)
                        self.error = True
                        continue
                    self.stateinfo[name] = statetype

    def get_rules(self):
        tsymbols = [f for f in self.ldict if f[:2] == 't_']
        self.toknames = {}
        self.funcsym = {}
        self.strsym = {}
        self.ignore = {}
        self.errorf = {}
        self.eoff = {}
        for s in self.stateinfo:
            self.funcsym[s] = []
            self.strsym[s] = []
        if len(tsymbols) == 0:
            self.log.error('No rules of the form t_rulename are defined')
            self.error = True
            return
        for f in tsymbols:
            t = self.ldict[f]
            states, tokname = _statetoken(f, self.stateinfo)
            self.toknames[f] = tokname
            if hasattr(t, '__call__'):
                if tokname == 'error':
                    for s in states:
                        self.errorf[s] = t
                elif tokname == 'eof':
                    for s in states:
                        self.eoff[s] = t
                elif tokname == 'ignore':
                    line = t.__code__.co_firstlineno
                    file = t.__code__.co_filename
                    self.log.error("%s:%d: Rule '%s' must be defined as a string", file, line, t.__name__)
                    self.error = True
                else:
                    for s in states:
                        self.funcsym[s].append((f, t))
            elif isinstance(t, StringTypes):
                if tokname == 'ignore':
                    for s in states:
                        self.ignore[s] = t
                    if '\\' in t:
                        self.log.warning("%s contains a literal backslash '\\'", f)
                elif tokname == 'error':
                    self.log.error("Rule '%s' must be defined as a function", f)
                    self.error = True
                else:
                    for s in states:
                        self.strsym[s].append((f, t))
            else:
                self.log.error('%s not defined as a function or string', f)
                self.error = True
        for f in self.funcsym.values():
            f.sort(key=lambda x: x[1].__code__.co_firstlineno)
        for s in self.strsym.values():
            s.sort(key=lambda x: len(x[1]), reverse=True)

    def validate_rules(self):
        for state in self.stateinfo:
            for fname, f in self.funcsym[state]:
                line = f.__code__.co_firstlineno
                file = f.__code__.co_filename
                module = inspect.getmodule(f)
                self.modules.add(module)
                tokname = self.toknames[fname]
                if isinstance(f, types.MethodType):
                    reqargs = 2
                else:
                    reqargs = 1
                nargs = f.__code__.co_argcount
                if nargs > reqargs:
                    self.log.error("%s:%d: Rule '%s' has too many arguments", file, line, f.__name__)
                    self.error = True
                    continue
                if nargs < reqargs:
                    self.log.error("%s:%d: Rule '%s' requires an argument", file, line, f.__name__)
                    self.error = True
                    continue
                if not _get_regex(f):
                    self.log.error("%s:%d: No regular expression defined for rule '%s'", file, line, f.__name__)
                    self.error = True
                    continue
                try:
                    c = re.compile('(?P<%s>%s)' % (fname, _get_regex(f)), self.reflags)
                    if c.match(''):
                        self.log.error("%s:%d: Regular expression for rule '%s' matches empty string", file, line, f.__name__)
                        self.error = True
                except re.error as e:
                    self.log.error("%s:%d: Invalid regular expression for rule '%s'. %s", file, line, f.__name__, e)
                    if '#' in _get_regex(f):
                        self.log.error("%s:%d. Make sure '#' in rule '%s' is escaped with '\\#'", file, line, f.__name__)
                    self.error = True
            for name, r in self.strsym[state]:
                tokname = self.toknames[name]
                if tokname == 'error':
                    self.log.error("Rule '%s' must be defined as a function", name)
                    self.error = True
                    continue
                if tokname not in self.tokens and tokname.find('ignore_') < 0:
                    self.log.error("Rule '%s' defined for an unspecified token %s", name, tokname)
                    self.error = True
                    continue
                try:
                    c = re.compile('(?P<%s>%s)' % (name, r), self.reflags)
                    if c.match(''):
                        self.log.error("Regular expression for rule '%s' matches empty string", name)
                        self.error = True
                except re.error as e:
                    self.log.error("Invalid regular expression for rule '%s'. %s", name, e)
                    if '#' in r:
                        self.log.error("Make sure '#' in rule '%s' is escaped with '\\#'", name)
                    self.error = True
            if not self.funcsym[state] and (not self.strsym[state]):
                self.log.error("No rules defined for state '%s'", state)
                self.error = True
            efunc = self.errorf.get(state, None)
            if efunc:
                f = efunc
                line = f.__code__.co_firstlineno
                file = f.__code__.co_filename
                module = inspect.getmodule(f)
                self.modules.add(module)
                if isinstance(f, types.MethodType):
                    reqargs = 2
                else:
                    reqargs = 1
                nargs = f.__code__.co_argcount
                if nargs > reqargs:
                    self.log.error("%s:%d: Rule '%s' has too many arguments", file, line, f.__name__)
                    self.error = True
                if nargs < reqargs:
                    self.log.error("%s:%d: Rule '%s' requires an argument", file, line, f.__name__)
                    self.error = True
        for module in self.modules:
            self.validate_module(module)

    def validate_module(self, module):
        try:
            lines, linen = inspect.getsourcelines(module)
        except IOError:
            return
        fre = re.compile('\\s*def\\s+(t_[a-zA-Z_0-9]*)\\(')
        sre = re.compile('\\s*(t_[a-zA-Z_0-9]*)\\s*=')
        counthash = {}
        linen += 1
        for line in lines:
            m = fre.match(line)
            if not m:
                m = sre.match(line)
            if m:
                name = m.group(1)
                prev = counthash.get(name)
                if not prev:
                    counthash[name] = linen
                else:
                    filename = inspect.getsourcefile(module)
                    self.log.error('%s:%d: Rule %s redefined. Previously defined on line %d', filename, linen, name, prev)
                    self.error = True
            linen += 1