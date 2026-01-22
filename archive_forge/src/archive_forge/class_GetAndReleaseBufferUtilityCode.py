from __future__ import absolute_import
from .Visitor import CythonTransform
from .ModuleNode import ModuleNode
from .Errors import CompileError
from .UtilityCode import CythonUtilityCode
from .Code import UtilityCode, TempitaUtilityCode
from . import Options
from . import Interpreter
from . import PyrexTypes
from . import Naming
from . import Symtab
class GetAndReleaseBufferUtilityCode(object):
    requires = None
    is_cython_utility = False

    def __init__(self):
        pass

    def __eq__(self, other):
        return isinstance(other, GetAndReleaseBufferUtilityCode)

    def __hash__(self):
        return 24342342

    def get_tree(self, **kwargs):
        pass

    def put_code(self, output):
        code = output['utility_code_def']
        proto_code = output['utility_code_proto']
        env = output.module_node.scope
        cython_scope = env.context.cython_scope
        types = []
        visited_scopes = set()

        def find_buffer_types(scope):
            if scope in visited_scopes:
                return
            visited_scopes.add(scope)
            for m in scope.cimported_modules:
                find_buffer_types(m)
            for e in scope.type_entries:
                if isinstance(e.utility_code_definition, CythonUtilityCode):
                    continue
                t = e.type
                if t.is_extension_type:
                    if scope is cython_scope and (not e.used):
                        continue
                    release = get = None
                    for x in t.scope.pyfunc_entries:
                        if x.name == u'__getbuffer__':
                            get = x.func_cname
                        elif x.name == u'__releasebuffer__':
                            release = x.func_cname
                    if get:
                        types.append((t.typeptr_cname, get, release))
        find_buffer_types(env)
        util_code = TempitaUtilityCode.load('GetAndReleaseBuffer', from_file='Buffer.c', context=dict(types=types))
        proto = util_code.format_code(util_code.proto)
        impl = util_code.format_code(util_code.inject_string_constants(util_code.impl, output)[1])
        proto_code.putln(proto)
        code.putln(impl)