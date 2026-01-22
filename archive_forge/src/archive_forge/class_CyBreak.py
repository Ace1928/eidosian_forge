from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
class CyBreak(CythonCommand):
    """
    Set a breakpoint for Cython code using Cython qualified name notation, e.g.:

        cy break cython_modulename.ClassName.method_name...

    or normal notation:

        cy break function_or_method_name...

    or for a line number:

        cy break cython_module:lineno...

    Set a Python breakpoint:
        Break on any function or method named 'func' in module 'modname'

            cy break -p modname.func...

        Break on any function or method named 'func'

            cy break -p func...
    """
    name = 'cy break'
    command_class = gdb.COMMAND_BREAKPOINTS

    def _break_pyx(self, name):
        modulename, _, lineno = name.partition(':')
        lineno = int(lineno)
        if modulename:
            cython_module = self.cy.cython_namespace[modulename]
        else:
            cython_module = self.get_cython_function().module
        if (cython_module.filename, lineno) in cython_module.lineno_cy2c:
            c_lineno = cython_module.lineno_cy2c[cython_module.filename, lineno]
            breakpoint = '%s:%s' % (cython_module.c_filename, c_lineno)
            gdb.execute('break ' + breakpoint)
        else:
            raise gdb.GdbError('Not a valid line number. Does it contain actual code?')

    def _break_funcname(self, funcname):
        func = self.cy.functions_by_qualified_name.get(funcname)
        if func and func.is_initmodule_function:
            func = None
        break_funcs = [func]
        if not func:
            funcs = self.cy.functions_by_name.get(funcname) or []
            funcs = [f for f in funcs if not f.is_initmodule_function]
            if not funcs:
                gdb.execute('break ' + funcname)
                return
            if len(funcs) > 1:
                print('There are multiple such functions:')
                for idx, func in enumerate(funcs):
                    print('%3d) %s' % (idx, func.qualified_name))
                while True:
                    try:
                        result = input("Select a function, press 'a' for all functions or press 'q' or '^D' to quit: ")
                    except EOFError:
                        return
                    else:
                        if result.lower() == 'q':
                            return
                        elif result.lower() == 'a':
                            break_funcs = funcs
                            break
                        elif result.isdigit() and 0 <= int(result) < len(funcs):
                            break_funcs = [funcs[int(result)]]
                            break
                        else:
                            print('Not understood...')
            else:
                break_funcs = [funcs[0]]
        for func in break_funcs:
            gdb.execute('break %s' % func.cname)
            if func.pf_cname:
                gdb.execute('break %s' % func.pf_cname)

    @libpython.dont_suppress_errors
    def invoke(self, function_names, from_tty):
        if isinstance(function_names, BYTES):
            function_names = function_names.decode(_filesystemencoding)
        argv = string_to_argv(function_names)
        if function_names.startswith('-p'):
            argv = argv[1:]
            python_breakpoints = True
        else:
            python_breakpoints = False
        for funcname in argv:
            if python_breakpoints:
                gdb.execute('py-break %s' % funcname)
            elif ':' in funcname:
                self._break_pyx(funcname)
            else:
                self._break_funcname(funcname)

    @libpython.dont_suppress_errors
    def complete(self, text, word):
        names = [n for n, L in self.cy.functions_by_name.items() if any((not f.is_initmodule_function for f in L))]
        qnames = [n for n, f in self.cy.functions_by_qualified_name.items() if not f.is_initmodule_function]
        if parameters.complete_unqualified:
            all_names = itertools.chain(qnames, names)
        else:
            all_names = qnames
        words = text.strip().split()
        if not words or '.' not in words[-1]:
            seen = set(text[:-len(word)].split())
            return [n for n in all_names if n.startswith(word) and n not in seen]
        lastword = words[-1]
        compl = [n for n in qnames if n.startswith(lastword)]
        if len(lastword) > len(word):
            strip_prefix_length = len(lastword) - len(word)
            compl = [n[strip_prefix_length:] for n in compl]
        return compl