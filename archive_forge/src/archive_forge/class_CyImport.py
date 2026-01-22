from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
class CyImport(CythonCommand):
    """
    Import debug information outputted by the Cython compiler
    Example: cy import FILE...
    """
    name = 'cy import'
    command_class = gdb.COMMAND_STATUS
    completer_class = gdb.COMPLETE_FILENAME

    @libpython.dont_suppress_errors
    def invoke(self, args, from_tty):
        if isinstance(args, BYTES):
            args = args.decode(_filesystemencoding)
        for arg in string_to_argv(args):
            try:
                f = open(arg)
            except OSError as e:
                raise gdb.GdbError('Unable to open file %r: %s' % (args, e.args[1]))
            t = etree.parse(f)
            for module in t.getroot():
                cython_module = CythonModule(**module.attrib)
                self.cy.cython_namespace[cython_module.name] = cython_module
                for variable in module.find('Globals'):
                    d = variable.attrib
                    cython_module.globals[d['name']] = CythonVariable(**d)
                for function in module.find('Functions'):
                    cython_function = CythonFunction(module=cython_module, **function.attrib)
                    name = cython_function.name
                    qname = cython_function.qualified_name
                    self.cy.functions_by_name[name].append(cython_function)
                    self.cy.functions_by_qualified_name[cython_function.qualified_name] = cython_function
                    self.cy.functions_by_cname[cython_function.cname] = cython_function
                    d = cython_module.functions[qname] = cython_function
                    for local in function.find('Locals'):
                        d = local.attrib
                        cython_function.locals[d['name']] = CythonVariable(**d)
                    for step_into_func in function.find('StepIntoFunctions'):
                        d = step_into_func.attrib
                        cython_function.step_into_functions.add(d['name'])
                    cython_function.arguments.extend((funcarg.tag for funcarg in function.find('Arguments')))
                for marker in module.find('LineNumberMapping'):
                    src_lineno = int(marker.attrib['src_lineno'])
                    src_path = marker.attrib['src_path']
                    c_linenos = list(map(int, marker.attrib['c_linenos'].split()))
                    cython_module.lineno_cy2c[src_path, src_lineno] = min(c_linenos)
                    for c_lineno in c_linenos:
                        cython_module.lineno_c2cy[c_lineno] = (src_path, src_lineno)