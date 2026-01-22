import __future__
import builtins
import importlib._bootstrap
import importlib._bootstrap_external
import importlib.machinery
import importlib.util
import inspect
import io
import os
import pkgutil
import platform
import re
import sys
import sysconfig
import time
import tokenize
import urllib.parse
import warnings
from collections import deque
from reprlib import Repr
from traceback import format_exception_only
class Helper:
    keywords = {'False': '', 'None': '', 'True': '', 'and': 'BOOLEAN', 'as': 'with', 'assert': ('assert', ''), 'async': ('async', ''), 'await': ('await', ''), 'break': ('break', 'while for'), 'class': ('class', 'CLASSES SPECIALMETHODS'), 'continue': ('continue', 'while for'), 'def': ('function', ''), 'del': ('del', 'BASICMETHODS'), 'elif': 'if', 'else': ('else', 'while for'), 'except': 'try', 'finally': 'try', 'for': ('for', 'break continue while'), 'from': 'import', 'global': ('global', 'nonlocal NAMESPACES'), 'if': ('if', 'TRUTHVALUE'), 'import': ('import', 'MODULES'), 'in': ('in', 'SEQUENCEMETHODS'), 'is': 'COMPARISON', 'lambda': ('lambda', 'FUNCTIONS'), 'nonlocal': ('nonlocal', 'global NAMESPACES'), 'not': 'BOOLEAN', 'or': 'BOOLEAN', 'pass': ('pass', ''), 'raise': ('raise', 'EXCEPTIONS'), 'return': ('return', 'FUNCTIONS'), 'try': ('try', 'EXCEPTIONS'), 'while': ('while', 'break continue if TRUTHVALUE'), 'with': ('with', 'CONTEXTMANAGERS EXCEPTIONS yield'), 'yield': ('yield', '')}
    _strprefixes = [p + q for p in ('b', 'f', 'r', 'u') for q in ("'", '"')]
    _symbols_inverse = {'STRINGS': ("'", "'''", '"', '"""', *_strprefixes), 'OPERATORS': ('+', '-', '*', '**', '/', '//', '%', '<<', '>>', '&', '|', '^', '~', '<', '>', '<=', '>=', '==', '!=', '<>'), 'COMPARISON': ('<', '>', '<=', '>=', '==', '!=', '<>'), 'UNARY': ('-', '~'), 'AUGMENTEDASSIGNMENT': ('+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>=', '**=', '//='), 'BITWISE': ('<<', '>>', '&', '|', '^', '~'), 'COMPLEX': ('j', 'J')}
    symbols = {'%': 'OPERATORS FORMATTING', '**': 'POWER', ',': 'TUPLES LISTS FUNCTIONS', '.': 'ATTRIBUTES FLOAT MODULES OBJECTS', '...': 'ELLIPSIS', ':': 'SLICINGS DICTIONARYLITERALS', '@': 'def class', '\\': 'STRINGS', '_': 'PRIVATENAMES', '__': 'PRIVATENAMES SPECIALMETHODS', '`': 'BACKQUOTES', '(': 'TUPLES FUNCTIONS CALLS', ')': 'TUPLES FUNCTIONS CALLS', '[': 'LISTS SUBSCRIPTS SLICINGS', ']': 'LISTS SUBSCRIPTS SLICINGS'}
    for topic, symbols_ in _symbols_inverse.items():
        for symbol in symbols_:
            topics = symbols.get(symbol, topic)
            if topic not in topics:
                topics = topics + ' ' + topic
            symbols[symbol] = topics
    del topic, symbols_, symbol, topics
    topics = {'TYPES': ('types', 'STRINGS UNICODE NUMBERS SEQUENCES MAPPINGS FUNCTIONS CLASSES MODULES FILES inspect'), 'STRINGS': ('strings', 'str UNICODE SEQUENCES STRINGMETHODS FORMATTING TYPES'), 'STRINGMETHODS': ('string-methods', 'STRINGS FORMATTING'), 'FORMATTING': ('formatstrings', 'OPERATORS'), 'UNICODE': ('strings', 'encodings unicode SEQUENCES STRINGMETHODS FORMATTING TYPES'), 'NUMBERS': ('numbers', 'INTEGER FLOAT COMPLEX TYPES'), 'INTEGER': ('integers', 'int range'), 'FLOAT': ('floating', 'float math'), 'COMPLEX': ('imaginary', 'complex cmath'), 'SEQUENCES': ('typesseq', 'STRINGMETHODS FORMATTING range LISTS'), 'MAPPINGS': 'DICTIONARIES', 'FUNCTIONS': ('typesfunctions', 'def TYPES'), 'METHODS': ('typesmethods', 'class def CLASSES TYPES'), 'CODEOBJECTS': ('bltin-code-objects', 'compile FUNCTIONS TYPES'), 'TYPEOBJECTS': ('bltin-type-objects', 'types TYPES'), 'FRAMEOBJECTS': 'TYPES', 'TRACEBACKS': 'TYPES', 'NONE': ('bltin-null-object', ''), 'ELLIPSIS': ('bltin-ellipsis-object', 'SLICINGS'), 'SPECIALATTRIBUTES': ('specialattrs', ''), 'CLASSES': ('types', 'class SPECIALMETHODS PRIVATENAMES'), 'MODULES': ('typesmodules', 'import'), 'PACKAGES': 'import', 'EXPRESSIONS': ('operator-summary', 'lambda or and not in is BOOLEAN COMPARISON BITWISE SHIFTING BINARY FORMATTING POWER UNARY ATTRIBUTES SUBSCRIPTS SLICINGS CALLS TUPLES LISTS DICTIONARIES'), 'OPERATORS': 'EXPRESSIONS', 'PRECEDENCE': 'EXPRESSIONS', 'OBJECTS': ('objects', 'TYPES'), 'SPECIALMETHODS': ('specialnames', 'BASICMETHODS ATTRIBUTEMETHODS CALLABLEMETHODS SEQUENCEMETHODS MAPPINGMETHODS NUMBERMETHODS CLASSES'), 'BASICMETHODS': ('customization', 'hash repr str SPECIALMETHODS'), 'ATTRIBUTEMETHODS': ('attribute-access', 'ATTRIBUTES SPECIALMETHODS'), 'CALLABLEMETHODS': ('callable-types', 'CALLS SPECIALMETHODS'), 'SEQUENCEMETHODS': ('sequence-types', 'SEQUENCES SEQUENCEMETHODS SPECIALMETHODS'), 'MAPPINGMETHODS': ('sequence-types', 'MAPPINGS SPECIALMETHODS'), 'NUMBERMETHODS': ('numeric-types', 'NUMBERS AUGMENTEDASSIGNMENT SPECIALMETHODS'), 'EXECUTION': ('execmodel', 'NAMESPACES DYNAMICFEATURES EXCEPTIONS'), 'NAMESPACES': ('naming', 'global nonlocal ASSIGNMENT DELETION DYNAMICFEATURES'), 'DYNAMICFEATURES': ('dynamic-features', ''), 'SCOPING': 'NAMESPACES', 'FRAMES': 'NAMESPACES', 'EXCEPTIONS': ('exceptions', 'try except finally raise'), 'CONVERSIONS': ('conversions', ''), 'IDENTIFIERS': ('identifiers', 'keywords SPECIALIDENTIFIERS'), 'SPECIALIDENTIFIERS': ('id-classes', ''), 'PRIVATENAMES': ('atom-identifiers', ''), 'LITERALS': ('atom-literals', 'STRINGS NUMBERS TUPLELITERALS LISTLITERALS DICTIONARYLITERALS'), 'TUPLES': 'SEQUENCES', 'TUPLELITERALS': ('exprlists', 'TUPLES LITERALS'), 'LISTS': ('typesseq-mutable', 'LISTLITERALS'), 'LISTLITERALS': ('lists', 'LISTS LITERALS'), 'DICTIONARIES': ('typesmapping', 'DICTIONARYLITERALS'), 'DICTIONARYLITERALS': ('dict', 'DICTIONARIES LITERALS'), 'ATTRIBUTES': ('attribute-references', 'getattr hasattr setattr ATTRIBUTEMETHODS'), 'SUBSCRIPTS': ('subscriptions', 'SEQUENCEMETHODS'), 'SLICINGS': ('slicings', 'SEQUENCEMETHODS'), 'CALLS': ('calls', 'EXPRESSIONS'), 'POWER': ('power', 'EXPRESSIONS'), 'UNARY': ('unary', 'EXPRESSIONS'), 'BINARY': ('binary', 'EXPRESSIONS'), 'SHIFTING': ('shifting', 'EXPRESSIONS'), 'BITWISE': ('bitwise', 'EXPRESSIONS'), 'COMPARISON': ('comparisons', 'EXPRESSIONS BASICMETHODS'), 'BOOLEAN': ('booleans', 'EXPRESSIONS TRUTHVALUE'), 'ASSERTION': 'assert', 'ASSIGNMENT': ('assignment', 'AUGMENTEDASSIGNMENT'), 'AUGMENTEDASSIGNMENT': ('augassign', 'NUMBERMETHODS'), 'DELETION': 'del', 'RETURNING': 'return', 'IMPORTING': 'import', 'CONDITIONAL': 'if', 'LOOPING': ('compound', 'for while break continue'), 'TRUTHVALUE': ('truth', 'if while and or not BASICMETHODS'), 'DEBUGGING': ('debugger', 'pdb'), 'CONTEXTMANAGERS': ('context-managers', 'with')}

    def __init__(self, input=None, output=None):
        self._input = input
        self._output = output

    @property
    def input(self):
        return self._input or sys.stdin

    @property
    def output(self):
        return self._output or sys.stdout

    def __repr__(self):
        if inspect.stack()[1][3] == '?':
            self()
            return ''
        return '<%s.%s instance>' % (self.__class__.__module__, self.__class__.__qualname__)
    _GoInteractive = object()

    def __call__(self, request=_GoInteractive):
        if request is not self._GoInteractive:
            try:
                self.help(request)
            except ImportError as e:
                self.output.write(f'{e}\n')
        else:
            self.intro()
            self.interact()
            self.output.write('\nYou are now leaving help and returning to the Python interpreter.\nIf you want to ask for help on a particular object directly from the\ninterpreter, you can type "help(object)".  Executing "help(\'string\')"\nhas the same effect as typing a particular string at the help> prompt.\n')

    def interact(self):
        self.output.write('\n')
        while True:
            try:
                request = self.getline('help> ')
                if not request:
                    break
            except (KeyboardInterrupt, EOFError):
                break
            request = request.strip()
            if len(request) > 2 and request[0] == request[-1] in ("'", '"') and (request[0] not in request[1:-1]):
                request = request[1:-1]
            if request.lower() in ('q', 'quit'):
                break
            if request == 'help':
                self.intro()
            else:
                self.help(request)

    def getline(self, prompt):
        """Read one line, using input() when appropriate."""
        if self.input is sys.stdin:
            return input(prompt)
        else:
            self.output.write(prompt)
            self.output.flush()
            return self.input.readline()

    def help(self, request, is_cli=False):
        if isinstance(request, str):
            request = request.strip()
            if request == 'keywords':
                self.listkeywords()
            elif request == 'symbols':
                self.listsymbols()
            elif request == 'topics':
                self.listtopics()
            elif request == 'modules':
                self.listmodules()
            elif request[:8] == 'modules ':
                self.listmodules(request.split()[1])
            elif request in self.symbols:
                self.showsymbol(request)
            elif request in ['True', 'False', 'None']:
                doc(eval(request), 'Help on %s:', is_cli=is_cli)
            elif request in self.keywords:
                self.showtopic(request)
            elif request in self.topics:
                self.showtopic(request)
            elif request:
                doc(request, 'Help on %s:', output=self._output, is_cli=is_cli)
            else:
                doc(str, 'Help on %s:', output=self._output, is_cli=is_cli)
        elif isinstance(request, Helper):
            self()
        else:
            doc(request, 'Help on %s:', output=self._output, is_cli=is_cli)
        self.output.write('\n')

    def intro(self):
        self.output.write('Welcome to Python {0}\'s help utility! If this is your first time using\nPython, you should definitely check out the tutorial at\nhttps://docs.python.org/{0}/tutorial/.\n\nEnter the name of any module, keyword, or topic to get help on writing\nPython programs and using Python modules.  To get a list of available\nmodules, keywords, symbols, or topics, enter "modules", "keywords",\n"symbols", or "topics".\n\nEach module also comes with a one-line summary of what it does; to list\nthe modules whose name or summary contain a given string such as "spam",\nenter "modules spam".\n\nTo quit this help utility and return to the interpreter,\nenter "q" or "quit".\n'.format('%d.%d' % sys.version_info[:2]))

    def list(self, items, columns=4, width=80):
        items = list(sorted(items))
        colw = width // columns
        rows = (len(items) + columns - 1) // columns
        for row in range(rows):
            for col in range(columns):
                i = col * rows + row
                if i < len(items):
                    self.output.write(items[i])
                    if col < columns - 1:
                        self.output.write(' ' + ' ' * (colw - 1 - len(items[i])))
            self.output.write('\n')

    def listkeywords(self):
        self.output.write('\nHere is a list of the Python keywords.  Enter any keyword to get more help.\n\n')
        self.list(self.keywords.keys())

    def listsymbols(self):
        self.output.write('\nHere is a list of the punctuation symbols which Python assigns special meaning\nto. Enter any symbol to get more help.\n\n')
        self.list(self.symbols.keys())

    def listtopics(self):
        self.output.write('\nHere is a list of available topics.  Enter any topic name to get more help.\n\n')
        self.list(self.topics.keys())

    def showtopic(self, topic, more_xrefs=''):
        try:
            import pydoc_data.topics
        except ImportError:
            self.output.write('\nSorry, topic and keyword documentation is not available because the\nmodule "pydoc_data.topics" could not be found.\n')
            return
        target = self.topics.get(topic, self.keywords.get(topic))
        if not target:
            self.output.write('no documentation found for %s\n' % repr(topic))
            return
        if type(target) is type(''):
            return self.showtopic(target, more_xrefs)
        label, xrefs = target
        try:
            doc = pydoc_data.topics.topics[label]
        except KeyError:
            self.output.write('no documentation found for %s\n' % repr(topic))
            return
        doc = doc.strip() + '\n'
        if more_xrefs:
            xrefs = (xrefs or '') + ' ' + more_xrefs
        if xrefs:
            import textwrap
            text = 'Related help topics: ' + ', '.join(xrefs.split()) + '\n'
            wrapped_text = textwrap.wrap(text, 72)
            doc += '\n%s\n' % '\n'.join(wrapped_text)
        pager(doc)

    def _gettopic(self, topic, more_xrefs=''):
        """Return unbuffered tuple of (topic, xrefs).

        If an error occurs here, the exception is caught and displayed by
        the url handler.

        This function duplicates the showtopic method but returns its
        result directly so it can be formatted for display in an html page.
        """
        try:
            import pydoc_data.topics
        except ImportError:
            return ('\nSorry, topic and keyword documentation is not available because the\nmodule "pydoc_data.topics" could not be found.\n', '')
        target = self.topics.get(topic, self.keywords.get(topic))
        if not target:
            raise ValueError('could not find topic')
        if isinstance(target, str):
            return self._gettopic(target, more_xrefs)
        label, xrefs = target
        doc = pydoc_data.topics.topics[label]
        if more_xrefs:
            xrefs = (xrefs or '') + ' ' + more_xrefs
        return (doc, xrefs)

    def showsymbol(self, symbol):
        target = self.symbols[symbol]
        topic, _, xrefs = target.partition(' ')
        self.showtopic(topic, xrefs)

    def listmodules(self, key=''):
        if key:
            self.output.write("\nHere is a list of modules whose name or summary contains '{}'.\nIf there are any, enter a module name to get more help.\n\n".format(key))
            apropos(key)
        else:
            self.output.write('\nPlease wait a moment while I gather a list of all available modules...\n\n')
            modules = {}

            def callback(path, modname, desc, modules=modules):
                if modname and modname[-9:] == '.__init__':
                    modname = modname[:-9] + ' (package)'
                if modname.find('.') < 0:
                    modules[modname] = 1

            def onerror(modname):
                callback(None, modname, None)
            ModuleScanner().run(callback, onerror=onerror)
            self.list(modules.keys())
            self.output.write('\nEnter any module name to get more help.  Or, type "modules spam" to search\nfor modules whose name or summary contain the string "spam".\n')