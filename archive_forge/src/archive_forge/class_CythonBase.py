from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
class CythonBase(object):

    @default_selected_gdb_frame(err=False)
    def is_cython_function(self, frame):
        return frame.name() in self.cy.functions_by_cname

    @default_selected_gdb_frame(err=False)
    def is_python_function(self, frame):
        """
        Tells if a frame is associated with a Python function.
        If we can't read the Python frame information, don't regard it as such.
        """
        if frame.name() == 'PyEval_EvalFrameEx':
            pyframe = libpython.Frame(frame).get_pyop()
            return pyframe and (not pyframe.is_optimized_out())
        return False

    @default_selected_gdb_frame()
    def get_c_function_name(self, frame):
        return frame.name()

    @default_selected_gdb_frame()
    def get_c_lineno(self, frame):
        return frame.find_sal().line

    @default_selected_gdb_frame()
    def get_cython_function(self, frame):
        result = self.cy.functions_by_cname.get(frame.name())
        if result is None:
            raise NoCythonFunctionInFrameError()
        return result

    @default_selected_gdb_frame()
    def get_cython_lineno(self, frame):
        """
        Get the current Cython line number. Returns 0 if there is no
        correspondence between the C and Cython code.
        """
        cyfunc = self.get_cython_function(frame)
        return cyfunc.module.lineno_c2cy.get(self.get_c_lineno(frame), 0)

    @default_selected_gdb_frame()
    def get_source_desc(self, frame):
        filename = lineno = lexer = None
        if self.is_cython_function(frame):
            filename = self.get_cython_function(frame).module.filename
            filename_and_lineno = self.get_cython_lineno(frame)
            assert filename == filename_and_lineno[0]
            lineno = filename_and_lineno[1]
            if pygments:
                lexer = pygments.lexers.CythonLexer(stripall=False)
        elif self.is_python_function(frame):
            pyframeobject = libpython.Frame(frame).get_pyop()
            if not pyframeobject:
                raise gdb.GdbError('Unable to read information on python frame')
            filename = pyframeobject.filename()
            lineno = pyframeobject.current_line_num()
            if pygments:
                lexer = pygments.lexers.PythonLexer(stripall=False)
        else:
            symbol_and_line_obj = frame.find_sal()
            if not symbol_and_line_obj or not symbol_and_line_obj.symtab:
                filename = None
                lineno = 0
            else:
                filename = symbol_and_line_obj.symtab.fullname()
                lineno = symbol_and_line_obj.line
                if pygments:
                    lexer = pygments.lexers.CLexer(stripall=False)
        return (SourceFileDescriptor(filename, lexer), lineno)

    @default_selected_gdb_frame()
    def get_source_line(self, frame):
        source_desc, lineno = self.get_source_desc()
        return source_desc.get_source(lineno)

    @default_selected_gdb_frame()
    def is_relevant_function(self, frame):
        """
        returns whether we care about a frame on the user-level when debugging
        Cython code
        """
        name = frame.name()
        older_frame = frame.older()
        if self.is_cython_function(frame) or self.is_python_function(frame):
            return True
        elif older_frame and self.is_cython_function(older_frame):
            cython_func = self.get_cython_function(older_frame)
            return name in cython_func.step_into_functions
        return False

    @default_selected_gdb_frame(err=False)
    def print_stackframe(self, frame, index, is_c=False):
        """
        Print a C, Cython or Python stack frame and the line of source code
        if available.
        """
        selected_frame = gdb.selected_frame()
        frame.select()
        try:
            source_desc, lineno = self.get_source_desc(frame)
        except NoFunctionNameInFrameError:
            print('#%-2d Unknown Frame (compile with -g)' % index)
            return
        if not is_c and self.is_python_function(frame):
            pyframe = libpython.Frame(frame).get_pyop()
            if pyframe is None or pyframe.is_optimized_out():
                return self.print_stackframe(frame, index, is_c=True)
            func_name = pyframe.co_name
            func_cname = 'PyEval_EvalFrameEx'
            func_args = []
        elif self.is_cython_function(frame):
            cyfunc = self.get_cython_function(frame)
            f = lambda arg: self.cy.cy_cvalue.invoke(arg, frame=frame)
            func_name = cyfunc.name
            func_cname = cyfunc.cname
            func_args = []
        else:
            source_desc, lineno = self.get_source_desc(frame)
            func_name = frame.name()
            func_cname = func_name
            func_args = []
        try:
            gdb_value = gdb.parse_and_eval(func_cname)
        except RuntimeError:
            func_address = 0
        else:
            func_address = gdb_value.address
            if not isinstance(func_address, int):
                if not isinstance(func_address, (str, bytes)):
                    func_address = str(func_address)
                func_address = int(func_address.split()[0], 0)
        a = ', '.join(('%s=%s' % (name, val) for name, val in func_args))
        sys.stdout.write('#%-2d 0x%016x in %s(%s)' % (index, func_address, func_name, a))
        if source_desc.filename is not None:
            sys.stdout.write(' at %s:%s' % (source_desc.filename, lineno))
        sys.stdout.write('\n')
        try:
            sys.stdout.write('    ' + source_desc.get_source(lineno))
        except gdb.GdbError:
            pass
        selected_frame.select()

    def get_remote_cython_globals_dict(self):
        m = gdb.parse_and_eval('__pyx_m')
        try:
            PyModuleObject = gdb.lookup_type('PyModuleObject')
        except RuntimeError:
            raise gdb.GdbError(textwrap.dedent('                Unable to lookup type PyModuleObject, did you compile python\n                with debugging support (-g)?'))
        m = m.cast(PyModuleObject.pointer())
        return m['md_dict']

    def get_cython_globals_dict(self):
        """
        Get the Cython globals dict where the remote names are turned into
        local strings.
        """
        remote_dict = self.get_remote_cython_globals_dict()
        pyobject_dict = libpython.PyObjectPtr.from_pyobject_ptr(remote_dict)
        result = {}
        seen = set()
        for k, v in pyobject_dict.iteritems():
            result[k.proxyval(seen)] = v
        return result

    def print_gdb_value(self, name, value, max_name_length=None, prefix=''):
        if libpython.pretty_printer_lookup(value):
            typename = ''
        else:
            typename = '(%s) ' % (value.type,)
        if max_name_length is None:
            print('%s%s = %s%s' % (prefix, name, typename, value))
        else:
            print('%s%-*s = %s%s' % (prefix, max_name_length, name, typename, value))

    def is_initialized(self, cython_func, local_name):
        cyvar = cython_func.locals[local_name]
        cur_lineno = self.get_cython_lineno()[1]
        if '->' in cyvar.cname:
            if cur_lineno > cython_func.lineno:
                if cyvar.type == PythonObject:
                    return int(gdb.parse_and_eval(cyvar.cname))
                return True
            return False
        return cur_lineno > cyvar.lineno