import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def _extern_python_decl(self, tp, name, tag_and_space):
    prnt = self._prnt
    if isinstance(tp.result, model.VoidType):
        size_of_result = '0'
    else:
        context = 'result of %s' % name
        size_of_result = '(int)sizeof(%s)' % (tp.result.get_c_name('', context),)
    prnt('static struct _cffi_externpy_s _cffi_externpy__%s =' % name)
    prnt('  { "%s.%s", %s, 0, 0 };' % (self.module_name, name, size_of_result))
    prnt()
    arguments = []
    context = 'argument of %s' % name
    for i, type in enumerate(tp.args):
        arg = type.get_c_name(' a%d' % i, context)
        arguments.append(arg)
    repr_arguments = ', '.join(arguments)
    repr_arguments = repr_arguments or 'void'
    name_and_arguments = '%s(%s)' % (name, repr_arguments)
    if tp.abi == '__stdcall':
        name_and_arguments = '_cffi_stdcall ' + name_and_arguments

    def may_need_128_bits(tp):
        return isinstance(tp, model.PrimitiveType) and tp.name == 'long double'
    size_of_a = max(len(tp.args) * 8, 8)
    if may_need_128_bits(tp.result):
        size_of_a = max(size_of_a, 16)
    if isinstance(tp.result, model.StructOrUnion):
        size_of_a = 'sizeof(%s) > %d ? sizeof(%s) : %d' % (tp.result.get_c_name(''), size_of_a, tp.result.get_c_name(''), size_of_a)
    prnt('%s%s' % (tag_and_space, tp.result.get_c_name(name_and_arguments)))
    prnt('{')
    prnt('  char a[%s];' % size_of_a)
    prnt('  char *p = a;')
    for i, type in enumerate(tp.args):
        arg = 'a%d' % i
        if isinstance(type, model.StructOrUnion) or may_need_128_bits(type):
            arg = '&' + arg
            type = model.PointerType(type)
        prnt('  *(%s)(p + %d) = %s;' % (type.get_c_name('*'), i * 8, arg))
    prnt('  _cffi_call_python(&_cffi_externpy__%s, p);' % name)
    if not isinstance(tp.result, model.VoidType):
        prnt('  return *(%s)p;' % (tp.result.get_c_name('*'),))
    prnt('}')
    prnt()
    self._num_externpy += 1