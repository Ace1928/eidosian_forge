from __future__ import absolute_import
import cython
import sys, copy
from itertools import chain
from . import Builtin
from .Errors import error, warning, InternalError, CompileError, CannotSpecialize
from . import Naming
from . import PyrexTypes
from . import TypeSlots
from .PyrexTypes import py_object_type, error_type
from .Symtab import (ModuleScope, LocalScope, ClosureScope, PropertyScope,
from .Code import UtilityCode
from .StringEncoding import EncodedString
from . import Future
from . import Options
from . import DebugFlags
from .Pythran import has_np_pythran, pythran_type, is_pythran_buffer
from ..Utils import add_metaclass, str_to_number
class DefNode(FuncDefNode):
    child_attrs = ['args', 'star_arg', 'starstar_arg', 'body', 'decorators', 'return_type_annotation']
    outer_attrs = ['decorators', 'return_type_annotation']
    is_staticmethod = False
    is_classmethod = False
    lambda_name = None
    reqd_kw_flags_cname = '0'
    is_wrapper = 0
    no_assignment_synthesis = 0
    decorators = None
    return_type_annotation = None
    entry = None
    acquire_gil = 0
    self_in_stararg = 0
    py_cfunc_node = None
    requires_classobj = False
    defaults_struct = None
    doc = None
    fused_py_func = False
    specialized_cpdefs = None
    py_wrapper = None
    py_wrapper_required = True
    func_cname = None
    defaults_getter = None

    def __init__(self, pos, **kwds):
        FuncDefNode.__init__(self, pos, **kwds)
        p = k = rk = r = 0
        for arg in self.args:
            if arg.pos_only:
                p += 1
            if arg.kw_only:
                k += 1
                if not arg.default:
                    rk += 1
            if not arg.default:
                r += 1
        self.num_posonly_args = p
        self.num_kwonly_args = k
        self.num_required_kw_args = rk
        self.num_required_args = r

    def as_cfunction(self, cfunc=None, scope=None, overridable=True, returns=None, except_val=None, has_explicit_exc_clause=False, modifiers=None, nogil=False, with_gil=False):
        if self.star_arg:
            error(self.star_arg.pos, 'cdef function cannot have star argument')
        if self.starstar_arg:
            error(self.starstar_arg.pos, 'cdef function cannot have starstar argument')
        exception_value, exception_check = except_val or (None, False)
        nogil = nogil or with_gil
        if cfunc is None:
            cfunc_args = []
            for formal_arg in self.args:
                name_declarator, type = formal_arg.analyse(scope, nonempty=1)
                cfunc_args.append(PyrexTypes.CFuncTypeArg(name=name_declarator.name, cname=None, annotation=formal_arg.annotation, type=py_object_type, pos=formal_arg.pos))
            cfunc_type = PyrexTypes.CFuncType(return_type=py_object_type, args=cfunc_args, has_varargs=False, exception_value=None, exception_check=exception_check, nogil=nogil, with_gil=with_gil, is_overridable=overridable)
            cfunc = CVarDefNode(self.pos, type=cfunc_type)
        else:
            if scope is None:
                scope = cfunc.scope
            cfunc_type = cfunc.type
            if len(self.args) != len(cfunc_type.args) or cfunc_type.has_varargs:
                error(self.pos, 'wrong number of arguments')
                error(cfunc.pos, 'previous declaration here')
            for i, (formal_arg, type_arg) in enumerate(zip(self.args, cfunc_type.args)):
                name_declarator, type = formal_arg.analyse(scope, nonempty=1, is_self_arg=i == 0 and scope.is_c_class_scope)
                if type is None or type is PyrexTypes.py_object_type:
                    formal_arg.type = type_arg.type
                    formal_arg.name_declarator = name_declarator
        if exception_value is None and cfunc_type.exception_value is not None:
            from .ExprNodes import ConstNode
            exception_value = ConstNode(self.pos, value=cfunc_type.exception_value, type=cfunc_type.return_type)
        declarator = CFuncDeclaratorNode(self.pos, base=CNameDeclaratorNode(self.pos, name=self.name, cname=None), args=self.args, has_varargs=False, exception_check=cfunc_type.exception_check, exception_value=exception_value, has_explicit_exc_clause=has_explicit_exc_clause, with_gil=cfunc_type.with_gil, nogil=cfunc_type.nogil)
        return CFuncDefNode(self.pos, modifiers=modifiers or [], base_type=CAnalysedBaseTypeNode(self.pos, type=cfunc_type.return_type), declarator=declarator, body=self.body, doc=self.doc, overridable=cfunc_type.is_overridable, type=cfunc_type, with_gil=cfunc_type.with_gil, nogil=cfunc_type.nogil, visibility='private', api=False, directive_locals=getattr(cfunc, 'directive_locals', {}), directive_returns=returns)

    def is_cdef_func_compatible(self):
        """Determines if the function's signature is compatible with a
        cdef function.  This can be used before calling
        .as_cfunction() to see if that will be successful.
        """
        if self.needs_closure:
            return False
        if self.star_arg or self.starstar_arg:
            return False
        return True

    def analyse_declarations(self, env):
        if self.decorators:
            for decorator in self.decorators:
                func = decorator.decorator
                if func.is_name:
                    self.is_classmethod |= func.name == 'classmethod'
                    self.is_staticmethod |= func.name == 'staticmethod'
        if self.is_classmethod and env.lookup_here('classmethod'):
            self.is_classmethod = False
        if self.is_staticmethod and env.lookup_here('staticmethod'):
            self.is_staticmethod = False
        if env.is_py_class_scope or env.is_c_class_scope:
            if self.name == '__new__' and env.is_py_class_scope:
                self.is_staticmethod = True
            elif self.name == '__init_subclass__' and env.is_c_class_scope:
                error(self.pos, "'__init_subclass__' is not supported by extension class")
            elif self.name in IMPLICIT_CLASSMETHODS and (not self.is_classmethod):
                self.is_classmethod = True
                from .ExprNodes import NameNode
                self.decorators = self.decorators or []
                self.decorators.insert(0, DecoratorNode(self.pos, decorator=NameNode(self.pos, name=EncodedString('classmethod'))))
        self.analyse_argument_types(env)
        if self.name == '<lambda>':
            self.declare_lambda_function(env)
        else:
            self.declare_pyfunction(env)
        self.analyse_signature(env)
        self.return_type = self.entry.signature.return_type()
        if self.return_type is py_object_type and self.return_type_annotation:
            if env.directives['annotation_typing'] and (not self.entry.is_special):
                _, return_type = self.return_type_annotation.analyse_type_annotation(env)
                if return_type and return_type.is_pyobject:
                    self.return_type = return_type
        self.create_local_scope(env)
        self.py_wrapper = DefNodeWrapper(self.pos, target=self, name=self.entry.name, args=self.args, star_arg=self.star_arg, starstar_arg=self.starstar_arg, return_type=self.return_type)
        self.py_wrapper.analyse_declarations(env)

    def analyse_argument_types(self, env):
        self.directive_locals = env.directives.get('locals', {})
        allow_none_for_extension_args = env.directives['allow_none_for_extension_args']
        f2s = env.fused_to_specific
        env.fused_to_specific = None
        for arg in self.args:
            if hasattr(arg, 'name'):
                name_declarator = None
            else:
                base_type = arg.base_type.analyse(env)
                if has_np_pythran(env) and base_type.is_pythran_expr:
                    base_type = PyrexTypes.FusedType([base_type, base_type.org_buffer])
                name_declarator, type = arg.declarator.analyse(base_type, env)
                arg.name = name_declarator.name
                arg.type = type
            self.align_argument_type(env, arg)
            if name_declarator and name_declarator.cname:
                error(self.pos, 'Python function argument cannot have C name specification')
            arg.type = arg.type.as_argument_type()
            arg.hdr_type = None
            arg.needs_conversion = 0
            arg.needs_type_test = 0
            arg.is_generic = 1
            if arg.type.is_pyobject or arg.type.is_buffer or arg.type.is_memoryviewslice:
                if arg.or_none:
                    arg.accept_none = True
                elif arg.not_none:
                    arg.accept_none = False
                elif arg.type.is_extension_type or arg.type.is_builtin_type or arg.type.is_buffer or arg.type.is_memoryviewslice:
                    if arg.default and arg.default.constant_result is None:
                        arg.accept_none = True
                    else:
                        arg.accept_none = allow_none_for_extension_args
                else:
                    arg.accept_none = True
            elif not arg.type.is_error:
                arg.accept_none = True
                if arg.not_none:
                    error(arg.pos, "Only Python type arguments can have 'not None'")
                if arg.or_none:
                    error(arg.pos, "Only Python type arguments can have 'or None'")
            if arg.type.is_fused:
                self.has_fused_arguments = True
        env.fused_to_specific = f2s
        if has_np_pythran(env):
            self.np_args_idx = [i for i, a in enumerate(self.args) if a.type.is_numpy_buffer]
        else:
            self.np_args_idx = []

    def analyse_signature(self, env):
        if self.entry.is_special:
            if self.decorators:
                error(self.pos, 'special functions of cdef classes cannot have decorators')
            self.entry.trivial_signature = len(self.args) == 1 and (not (self.star_arg or self.starstar_arg))
        elif not (self.star_arg or self.starstar_arg) and (not env.directives['always_allow_keywords'] or all([arg.pos_only for arg in self.args])):
            if self.entry.signature is TypeSlots.pyfunction_signature:
                if len(self.args) == 0:
                    self.entry.signature = TypeSlots.pyfunction_noargs
                elif len(self.args) == 1:
                    if self.args[0].default is None and (not self.args[0].kw_only):
                        self.entry.signature = TypeSlots.pyfunction_onearg
            elif self.entry.signature is TypeSlots.pymethod_signature:
                if len(self.args) == 1:
                    self.entry.signature = TypeSlots.unaryfunc
                elif len(self.args) == 2:
                    if self.args[1].default is None and (not self.args[1].kw_only):
                        self.entry.signature = TypeSlots.ibinaryfunc
        sig = self.entry.signature
        nfixed = sig.max_num_fixed_args()
        min_nfixed = sig.min_num_fixed_args()
        if sig is TypeSlots.pymethod_signature and nfixed == 1 and (len(self.args) == 0) and self.star_arg:
            sig = self.entry.signature = TypeSlots.pyfunction_signature
            self.self_in_stararg = 1
            nfixed = min_nfixed = 0
        if self.is_staticmethod and env.is_c_class_scope:
            nfixed = min_nfixed = 0
            self.self_in_stararg = True
            self.entry.signature = sig = copy.copy(sig)
            sig.fixed_arg_format = '*'
            sig.is_staticmethod = True
            sig.has_generic_args = True
        if (self.is_classmethod or self.is_staticmethod) and self.has_fused_arguments and env.is_c_class_scope:
            del self.decorator_indirection.stats[:]
        for i in range(min(nfixed, len(self.args))):
            arg = self.args[i]
            arg.is_generic = 0
            if i >= min_nfixed:
                arg.is_special_method_optional = True
            if sig.is_self_arg(i) and (not self.is_staticmethod):
                if self.is_classmethod:
                    arg.is_type_arg = 1
                    arg.hdr_type = arg.type = Builtin.type_type
                else:
                    arg.is_self_arg = 1
                    arg.hdr_type = arg.type = env.parent_type
                arg.needs_conversion = 0
            else:
                arg.hdr_type = sig.fixed_arg_type(i)
                if not arg.type.same_as(arg.hdr_type):
                    if arg.hdr_type.is_pyobject and arg.type.is_pyobject:
                        arg.needs_type_test = 1
                    else:
                        arg.needs_conversion = 1
        if min_nfixed > len(self.args):
            self.bad_signature()
            return
        elif nfixed < len(self.args):
            if not sig.has_generic_args:
                self.bad_signature()
            for arg in self.args:
                if arg.is_generic and (arg.type.is_extension_type or arg.type.is_builtin_type):
                    arg.needs_type_test = 1
        mf = sig.method_flags()
        if mf and TypeSlots.method_varargs in mf and (not self.entry.is_special):
            if self.star_arg:
                uses_args_tuple = True
                for arg in self.args:
                    if arg.is_generic and (not arg.kw_only) and (not arg.is_self_arg) and (not arg.is_type_arg):
                        uses_args_tuple = False
            else:
                uses_args_tuple = False
            if not uses_args_tuple:
                sig = self.entry.signature = sig.with_fastcall()

    def bad_signature(self):
        sig = self.entry.signature
        expected_str = '%d' % sig.min_num_fixed_args()
        if sig.has_generic_args:
            expected_str += ' or more'
        elif sig.optional_object_arg_count:
            expected_str += ' to %d' % sig.max_num_fixed_args()
        name = self.name
        if name.startswith('__') and name.endswith('__'):
            desc = 'Special method'
        else:
            desc = 'Method'
        error(self.pos, '%s %s has wrong number of arguments (%d declared, %s expected)' % (desc, self.name, len(self.args), expected_str))

    def declare_pyfunction(self, env):
        name = self.name
        entry = env.lookup_here(name)
        if entry:
            if entry.is_final_cmethod and (not env.parent_type.is_final_type):
                error(self.pos, 'Only final types can have final Python (def/cpdef) methods')
            if entry.type.is_cfunction and (not entry.is_builtin_cmethod) and (not self.is_wrapper):
                warning(self.pos, 'Overriding cdef method with def method.', 5)
        entry = env.declare_pyfunction(name, self.pos, allow_redefine=not self.is_wrapper)
        self.entry = entry
        prefix = env.next_id(env.scope_prefix)
        self.entry.pyfunc_cname = punycodify_name(Naming.pyfunc_prefix + prefix + name)
        if Options.docstrings:
            entry.doc = embed_position(self.pos, self.doc)
            entry.doc_cname = punycodify_name(Naming.funcdoc_prefix + prefix + name)
            if entry.is_special:
                if entry.name in TypeSlots.invisible or not entry.doc or (entry.name in '__getattr__' and env.directives['fast_getattr']):
                    entry.wrapperbase_cname = None
                else:
                    entry.wrapperbase_cname = punycodify_name(Naming.wrapperbase_prefix + prefix + name)
        else:
            entry.doc = None

    def declare_lambda_function(self, env):
        entry = env.declare_lambda_function(self.lambda_name, self.pos)
        entry.doc = None
        self.entry = entry
        self.entry.pyfunc_cname = entry.cname

    def declare_arguments(self, env):
        for arg in self.args:
            if not arg.name:
                error(arg.pos, 'Missing argument name')
            if arg.needs_conversion:
                arg.entry = env.declare_var(arg.name, arg.type, arg.pos)
                if arg.type.is_pyobject:
                    arg.entry.init = '0'
            else:
                arg.entry = self.declare_argument(env, arg)
            arg.entry.is_arg = 1
            arg.entry.used = 1
            arg.entry.is_self_arg = arg.is_self_arg
        self.declare_python_arg(env, self.star_arg)
        self.declare_python_arg(env, self.starstar_arg)

    def declare_python_arg(self, env, arg):
        if arg:
            if env.directives['infer_types'] != False:
                type = PyrexTypes.unspecified_type
            else:
                type = py_object_type
            entry = env.declare_var(arg.name, type, arg.pos)
            entry.is_arg = 1
            entry.used = 1
            entry.init = '0'
            entry.xdecref_cleanup = 1
            arg.entry = entry

    def analyse_expressions(self, env):
        self.local_scope.directives = env.directives
        self.analyse_default_values(env)
        self.analyse_annotations(env)
        if not self.needs_assignment_synthesis(env) and self.decorators:
            for decorator in self.decorators[::-1]:
                decorator.decorator = decorator.decorator.analyse_expressions(env)
        self.py_wrapper.prepare_argument_coercion(env)
        return self

    def needs_assignment_synthesis(self, env, code=None):
        if self.is_staticmethod:
            return True
        if self.specialized_cpdefs or self.entry.is_fused_specialized:
            return False
        if self.no_assignment_synthesis:
            return False
        if self.entry.is_special:
            return False
        if self.entry.is_anonymous:
            return True
        if env.is_module_scope or env.is_c_class_scope:
            if code is None:
                return self.local_scope.directives['binding']
            else:
                return code.globalstate.directives['binding']
        return env.is_py_class_scope or env.is_closure_scope

    def error_value(self):
        return self.entry.signature.error_value

    def caller_will_check_exceptions(self):
        return self.entry.signature.exception_check

    def generate_function_definitions(self, env, code):
        if self.defaults_getter:
            self.defaults_getter.generate_function_definitions(env.global_scope(), code)
        if self.py_wrapper_required:
            self.py_wrapper.func_cname = self.entry.func_cname
            self.py_wrapper.generate_function_definitions(env, code)
        FuncDefNode.generate_function_definitions(self, env, code)

    def generate_function_header(self, code, with_pymethdef, proto_only=0):
        if proto_only:
            if self.py_wrapper_required:
                self.py_wrapper.generate_function_header(code, with_pymethdef, True)
            return
        arg_code_list = []
        if self.entry.signature.has_dummy_arg:
            self_arg = 'PyObject *%s' % Naming.self_cname
            if not self.needs_outer_scope:
                self_arg = 'CYTHON_UNUSED ' + self_arg
            arg_code_list.append(self_arg)

        def arg_decl_code(arg):
            entry = arg.entry
            if entry.in_closure:
                cname = entry.original_cname
            else:
                cname = entry.cname
            decl = entry.type.declaration_code(cname)
            if not entry.cf_used:
                decl = 'CYTHON_UNUSED ' + decl
            return decl
        for arg in self.args:
            arg_code_list.append(arg_decl_code(arg))
        if self.star_arg:
            arg_code_list.append(arg_decl_code(self.star_arg))
        if self.starstar_arg:
            arg_code_list.append(arg_decl_code(self.starstar_arg))
        if arg_code_list:
            arg_code = ', '.join(arg_code_list)
        else:
            arg_code = 'void'
        dc = self.return_type.declaration_code(self.entry.pyfunc_cname)
        decls_code = code.globalstate['decls']
        preprocessor_guard = self.get_preprocessor_guard()
        if preprocessor_guard:
            decls_code.putln(preprocessor_guard)
        decls_code.putln('static %s(%s); /* proto */' % (dc, arg_code))
        if preprocessor_guard:
            decls_code.putln('#endif')
        code.putln('static %s(%s) {' % (dc, arg_code))

    def generate_argument_declarations(self, env, code):
        pass

    def generate_keyword_list(self, code):
        pass

    def generate_argument_parsing_code(self, env, code):

        def put_into_closure(entry):
            if entry.in_closure:
                if entry.type.is_array:
                    assert entry.type.size is not None
                    code.globalstate.use_utility_code(UtilityCode.load_cached('IncludeStringH', 'StringTools.c'))
                    code.putln('memcpy({0}, {1}, sizeof({0}));'.format(entry.cname, entry.original_cname))
                else:
                    code.putln('%s = %s;' % (entry.cname, entry.original_cname))
                if entry.type.is_memoryviewslice:
                    entry.type.generate_incref_memoryviewslice(code, entry.cname, True)
                elif entry.xdecref_cleanup:
                    code.put_var_xincref(entry)
                    code.put_var_xgiveref(entry)
                else:
                    code.put_var_incref(entry)
                    code.put_var_giveref(entry)
        for arg in self.args:
            put_into_closure(arg.entry)
        for arg in (self.star_arg, self.starstar_arg):
            if arg:
                put_into_closure(arg.entry)

    def generate_argument_type_tests(self, code):
        pass