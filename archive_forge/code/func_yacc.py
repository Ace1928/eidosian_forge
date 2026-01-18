import re
import types
import sys
import os.path
import inspect
import base64
import warnings
def yacc(method='LALR', debug=yaccdebug, module=None, tabmodule=tab_module, start=None, check_recursion=True, optimize=False, write_tables=True, debugfile=debug_file, outputdir=None, debuglog=None, errorlog=None, picklefile=None):
    if tabmodule is None:
        tabmodule = tab_module
    global parse
    if picklefile:
        write_tables = 0
    if errorlog is None:
        errorlog = PlyLogger(sys.stderr)
    if module:
        _items = [(k, getattr(module, k)) for k in dir(module)]
        pdict = dict(_items)
        if '__file__' not in pdict:
            pdict['__file__'] = sys.modules[pdict['__module__']].__file__
    else:
        pdict = get_caller_module_dict(2)
    if outputdir is None:
        if isinstance(tabmodule, types.ModuleType):
            srcfile = tabmodule.__file__
        elif '.' not in tabmodule:
            srcfile = pdict['__file__']
        else:
            parts = tabmodule.split('.')
            pkgname = '.'.join(parts[:-1])
            exec('import %s' % pkgname)
            srcfile = getattr(sys.modules[pkgname], '__file__', '')
        outputdir = os.path.dirname(srcfile)
    pkg = pdict.get('__package__')
    if pkg and isinstance(tabmodule, str):
        if '.' not in tabmodule:
            tabmodule = pkg + '.' + tabmodule
    if start is not None:
        pdict['start'] = start
    pinfo = ParserReflect(pdict, log=errorlog)
    pinfo.get_all()
    if pinfo.error:
        raise YaccError('Unable to build parser')
    signature = pinfo.signature()
    try:
        lr = LRTable()
        if picklefile:
            read_signature = lr.read_pickle(picklefile)
        else:
            read_signature = lr.read_table(tabmodule)
        if optimize or read_signature == signature:
            try:
                lr.bind_callables(pinfo.pdict)
                parser = LRParser(lr, pinfo.error_func)
                parse = parser.parse
                return parser
            except Exception as e:
                errorlog.warning('There was a problem loading the table file: %r', e)
    except VersionError as e:
        errorlog.warning(str(e))
    except ImportError:
        pass
    if debuglog is None:
        if debug:
            try:
                debuglog = PlyLogger(open(os.path.join(outputdir, debugfile), 'w'))
            except IOError as e:
                errorlog.warning("Couldn't open %r. %s" % (debugfile, e))
                debuglog = NullLogger()
        else:
            debuglog = NullLogger()
    debuglog.info('Created by PLY version %s (http://www.dabeaz.com/ply)', __version__)
    errors = False
    if pinfo.validate_all():
        raise YaccError('Unable to build parser')
    if not pinfo.error_func:
        errorlog.warning('no p_error() function is defined')
    grammar = Grammar(pinfo.tokens)
    for term, assoc, level in pinfo.preclist:
        try:
            grammar.set_precedence(term, assoc, level)
        except GrammarError as e:
            errorlog.warning('%s', e)
    for funcname, gram in pinfo.grammar:
        file, line, prodname, syms = gram
        try:
            grammar.add_production(prodname, syms, funcname, file, line)
        except GrammarError as e:
            errorlog.error('%s', e)
            errors = True
    try:
        if start is None:
            grammar.set_start(pinfo.start)
        else:
            grammar.set_start(start)
    except GrammarError as e:
        errorlog.error(str(e))
        errors = True
    if errors:
        raise YaccError('Unable to build parser')
    undefined_symbols = grammar.undefined_symbols()
    for sym, prod in undefined_symbols:
        errorlog.error('%s:%d: Symbol %r used, but not defined as a token or a rule', prod.file, prod.line, sym)
        errors = True
    unused_terminals = grammar.unused_terminals()
    if unused_terminals:
        debuglog.info('')
        debuglog.info('Unused terminals:')
        debuglog.info('')
        for term in unused_terminals:
            errorlog.warning('Token %r defined, but not used', term)
            debuglog.info('    %s', term)
    if debug:
        debuglog.info('')
        debuglog.info('Grammar')
        debuglog.info('')
        for n, p in enumerate(grammar.Productions):
            debuglog.info('Rule %-5d %s', n, p)
    unused_rules = grammar.unused_rules()
    for prod in unused_rules:
        errorlog.warning('%s:%d: Rule %r defined, but not used', prod.file, prod.line, prod.name)
    if len(unused_terminals) == 1:
        errorlog.warning('There is 1 unused token')
    if len(unused_terminals) > 1:
        errorlog.warning('There are %d unused tokens', len(unused_terminals))
    if len(unused_rules) == 1:
        errorlog.warning('There is 1 unused rule')
    if len(unused_rules) > 1:
        errorlog.warning('There are %d unused rules', len(unused_rules))
    if debug:
        debuglog.info('')
        debuglog.info('Terminals, with rules where they appear')
        debuglog.info('')
        terms = list(grammar.Terminals)
        terms.sort()
        for term in terms:
            debuglog.info('%-20s : %s', term, ' '.join([str(s) for s in grammar.Terminals[term]]))
        debuglog.info('')
        debuglog.info('Nonterminals, with rules where they appear')
        debuglog.info('')
        nonterms = list(grammar.Nonterminals)
        nonterms.sort()
        for nonterm in nonterms:
            debuglog.info('%-20s : %s', nonterm, ' '.join([str(s) for s in grammar.Nonterminals[nonterm]]))
        debuglog.info('')
    if check_recursion:
        unreachable = grammar.find_unreachable()
        for u in unreachable:
            errorlog.warning('Symbol %r is unreachable', u)
        infinite = grammar.infinite_cycles()
        for inf in infinite:
            errorlog.error('Infinite recursion detected for symbol %r', inf)
            errors = True
    unused_prec = grammar.unused_precedence()
    for term, assoc in unused_prec:
        errorlog.error('Precedence rule %r defined for unknown symbol %r', assoc, term)
        errors = True
    if errors:
        raise YaccError('Unable to build parser')
    if debug:
        errorlog.debug('Generating %s tables', method)
    lr = LRGeneratedTable(grammar, method, debuglog)
    if debug:
        num_sr = len(lr.sr_conflicts)
        if num_sr == 1:
            errorlog.warning('1 shift/reduce conflict')
        elif num_sr > 1:
            errorlog.warning('%d shift/reduce conflicts', num_sr)
        num_rr = len(lr.rr_conflicts)
        if num_rr == 1:
            errorlog.warning('1 reduce/reduce conflict')
        elif num_rr > 1:
            errorlog.warning('%d reduce/reduce conflicts', num_rr)
    if debug and (lr.sr_conflicts or lr.rr_conflicts):
        debuglog.warning('')
        debuglog.warning('Conflicts:')
        debuglog.warning('')
        for state, tok, resolution in lr.sr_conflicts:
            debuglog.warning('shift/reduce conflict for %s in state %d resolved as %s', tok, state, resolution)
        already_reported = set()
        for state, rule, rejected in lr.rr_conflicts:
            if (state, id(rule), id(rejected)) in already_reported:
                continue
            debuglog.warning('reduce/reduce conflict in state %d resolved using rule (%s)', state, rule)
            debuglog.warning('rejected rule (%s) in state %d', rejected, state)
            errorlog.warning('reduce/reduce conflict in state %d resolved using rule (%s)', state, rule)
            errorlog.warning('rejected rule (%s) in state %d', rejected, state)
            already_reported.add((state, id(rule), id(rejected)))
        warned_never = []
        for state, rule, rejected in lr.rr_conflicts:
            if not rejected.reduced and rejected not in warned_never:
                debuglog.warning('Rule (%s) is never reduced', rejected)
                errorlog.warning('Rule (%s) is never reduced', rejected)
                warned_never.append(rejected)
    if write_tables:
        try:
            lr.write_table(tabmodule, outputdir, signature)
        except IOError as e:
            errorlog.warning("Couldn't create %r. %s" % (tabmodule, e))
    if picklefile:
        try:
            lr.pickle_table(picklefile, signature)
        except IOError as e:
            errorlog.warning("Couldn't create %r. %s" % (picklefile, e))
    lr.bind_callables(pinfo.pdict)
    parser = LRParser(lr, pinfo.error_func)
    parse = parser.parse
    return parser