import dis
import inspect
import opcode as _opcode
import struct
import sys
import types
from _pydevd_frame_eval.vendored import bytecode as _bytecode
from _pydevd_frame_eval.vendored.bytecode.instr import (
class ConcreteBytecode(_bytecode._BaseBytecodeList):

    def __init__(self, instructions=(), *, consts=(), names=(), varnames=()):
        super().__init__()
        self.consts = list(consts)
        self.names = list(names)
        self.varnames = list(varnames)
        for instr in instructions:
            self._check_instr(instr)
        self.extend(instructions)

    def __iter__(self):
        instructions = super().__iter__()
        for instr in instructions:
            self._check_instr(instr)
            yield instr

    def _check_instr(self, instr):
        if not isinstance(instr, (ConcreteInstr, SetLineno)):
            raise ValueError('ConcreteBytecode must only contain ConcreteInstr and SetLineno objects, but %s was found' % type(instr).__name__)

    def _copy_attr_from(self, bytecode):
        super()._copy_attr_from(bytecode)
        if isinstance(bytecode, ConcreteBytecode):
            self.consts = bytecode.consts
            self.names = bytecode.names
            self.varnames = bytecode.varnames

    def __repr__(self):
        return '<ConcreteBytecode instr#=%s>' % len(self)

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        const_keys1 = list(map(const_key, self.consts))
        const_keys2 = list(map(const_key, other.consts))
        if const_keys1 != const_keys2:
            return False
        if self.names != other.names:
            return False
        if self.varnames != other.varnames:
            return False
        return super().__eq__(other)

    @staticmethod
    def from_code(code, *, extended_arg=False):
        line_starts = dict(dis.findlinestarts(code))
        instructions = []
        offset = 0
        lineno = code.co_firstlineno
        while offset < len(code.co_code) // (2 if OFFSET_AS_INSTRUCTION else 1):
            lineno_off = 2 * offset if OFFSET_AS_INSTRUCTION else offset
            if lineno_off in line_starts:
                lineno = line_starts[lineno_off]
            instr = ConcreteInstr.disassemble(lineno, code.co_code, offset)
            instructions.append(instr)
            offset += instr.size // 2 if OFFSET_AS_INSTRUCTION else instr.size
        bytecode = ConcreteBytecode()
        if not extended_arg:
            bytecode._remove_extended_args(instructions)
        bytecode.name = code.co_name
        bytecode.filename = code.co_filename
        bytecode.flags = code.co_flags
        bytecode.argcount = code.co_argcount
        if sys.version_info >= (3, 8):
            bytecode.posonlyargcount = code.co_posonlyargcount
        bytecode.kwonlyargcount = code.co_kwonlyargcount
        bytecode.first_lineno = code.co_firstlineno
        bytecode.names = list(code.co_names)
        bytecode.consts = list(code.co_consts)
        bytecode.varnames = list(code.co_varnames)
        bytecode.freevars = list(code.co_freevars)
        bytecode.cellvars = list(code.co_cellvars)
        _set_docstring(bytecode, code.co_consts)
        bytecode[:] = instructions
        return bytecode

    @staticmethod
    def _normalize_lineno(instructions, first_lineno):
        lineno = first_lineno
        for instr in instructions:
            if instr.lineno is not None:
                lineno = instr.lineno
            if isinstance(instr, ConcreteInstr):
                yield (lineno, instr)

    def _assemble_code(self):
        offset = 0
        code_str = []
        linenos = []
        for lineno, instr in self._normalize_lineno(self, self.first_lineno):
            code_str.append(instr.assemble())
            i_size = instr.size
            linenos.append((offset * 2 if OFFSET_AS_INSTRUCTION else offset, i_size, lineno))
            offset += i_size // 2 if OFFSET_AS_INSTRUCTION else i_size
        code_str = b''.join(code_str)
        return (code_str, linenos)

    @staticmethod
    def _assemble_lnotab(first_lineno, linenos):
        lnotab = []
        old_offset = 0
        old_lineno = first_lineno
        for offset, _, lineno in linenos:
            dlineno = lineno - old_lineno
            if dlineno == 0:
                continue
            if dlineno < 0 and sys.version_info < (3, 6):
                raise ValueError('negative line number delta is not supported on Python < 3.6')
            old_lineno = lineno
            doff = offset - old_offset
            old_offset = offset
            while doff > 255:
                lnotab.append(b'\xff\x00')
                doff -= 255
            while dlineno < -128:
                lnotab.append(struct.pack('Bb', doff, -128))
                doff = 0
                dlineno -= -128
            while dlineno > 127:
                lnotab.append(struct.pack('Bb', doff, 127))
                doff = 0
                dlineno -= 127
            assert 0 <= doff <= 255
            assert -128 <= dlineno <= 127
            lnotab.append(struct.pack('Bb', doff, dlineno))
        return b''.join(lnotab)

    @staticmethod
    def _pack_linetable(doff, dlineno, linetable):
        while dlineno < -127:
            linetable.append(struct.pack('Bb', 0, -127))
            dlineno -= -127
        while dlineno > 127:
            linetable.append(struct.pack('Bb', 0, 127))
            dlineno -= 127
        if doff > 254:
            linetable.append(struct.pack('Bb', 254, dlineno))
            doff -= 254
            while doff > 254:
                linetable.append(b'\xfe\x00')
                doff -= 254
            linetable.append(struct.pack('Bb', doff, 0))
        else:
            linetable.append(struct.pack('Bb', doff, dlineno))
        assert 0 <= doff <= 254
        assert -127 <= dlineno <= 127

    def _assemble_linestable(self, first_lineno, linenos):
        if not linenos:
            return b''
        linetable = []
        old_offset = 0
        iter_in = iter(linenos)
        offset, i_size, old_lineno = next(iter_in)
        old_dlineno = old_lineno - first_lineno
        for offset, i_size, lineno in iter_in:
            dlineno = lineno - old_lineno
            if dlineno == 0:
                continue
            old_lineno = lineno
            doff = offset - old_offset
            old_offset = offset
            self._pack_linetable(doff, old_dlineno, linetable)
            old_dlineno = dlineno
        doff = offset + i_size - old_offset
        self._pack_linetable(doff, old_dlineno, linetable)
        return b''.join(linetable)

    @staticmethod
    def _remove_extended_args(instructions):
        nb_extended_args = 0
        extended_arg = None
        index = 0
        while index < len(instructions):
            instr = instructions[index]
            if isinstance(instr, SetLineno):
                index += 1
                continue
            if instr.name == 'EXTENDED_ARG':
                nb_extended_args += 1
                if extended_arg is not None:
                    extended_arg = (extended_arg << 8) + instr.arg
                else:
                    extended_arg = instr.arg
                del instructions[index]
                continue
            if extended_arg is not None:
                arg = (extended_arg << 8) + instr.arg
                extended_arg = None
                instr = ConcreteInstr(instr.name, arg, lineno=instr.lineno, extended_args=nb_extended_args, offset=instr.offset)
                instructions[index] = instr
                nb_extended_args = 0
            index += 1
        if extended_arg is not None:
            raise ValueError('EXTENDED_ARG at the end of the code')

    def compute_stacksize(self, *, check_pre_and_post=True):
        bytecode = self.to_bytecode()
        cfg = _bytecode.ControlFlowGraph.from_bytecode(bytecode)
        return cfg.compute_stacksize(check_pre_and_post=check_pre_and_post)

    def to_code(self, stacksize=None, *, check_pre_and_post=True):
        code_str, linenos = self._assemble_code()
        lnotab = self._assemble_linestable(self.first_lineno, linenos) if sys.version_info >= (3, 10) else self._assemble_lnotab(self.first_lineno, linenos)
        nlocals = len(self.varnames)
        if stacksize is None:
            stacksize = self.compute_stacksize(check_pre_and_post=check_pre_and_post)
        if sys.version_info < (3, 8):
            return types.CodeType(self.argcount, self.kwonlyargcount, nlocals, stacksize, int(self.flags), code_str, tuple(self.consts), tuple(self.names), tuple(self.varnames), self.filename, self.name, self.first_lineno, lnotab, tuple(self.freevars), tuple(self.cellvars))
        else:
            return types.CodeType(self.argcount, self.posonlyargcount, self.kwonlyargcount, nlocals, stacksize, int(self.flags), code_str, tuple(self.consts), tuple(self.names), tuple(self.varnames), self.filename, self.name, self.first_lineno, lnotab, tuple(self.freevars), tuple(self.cellvars))

    def to_bytecode(self):
        c_instructions = self[:]
        self._remove_extended_args(c_instructions)
        jump_targets = set()
        offset = 0
        for instr in c_instructions:
            if isinstance(instr, SetLineno):
                continue
            target = instr.get_jump_target(offset)
            if target is not None:
                jump_targets.add(target)
            offset += instr.size // 2 if OFFSET_AS_INSTRUCTION else instr.size
        jumps = []
        instructions = []
        labels = {}
        offset = 0
        ncells = len(self.cellvars)
        for lineno, instr in self._normalize_lineno(c_instructions, self.first_lineno):
            if offset in jump_targets:
                label = Label()
                labels[offset] = label
                instructions.append(label)
            jump_target = instr.get_jump_target(offset)
            size = instr.size
            arg = instr.arg
            if instr.opcode in _opcode.hasconst:
                arg = self.consts[arg]
            elif instr.opcode in _opcode.haslocal:
                arg = self.varnames[arg]
            elif instr.opcode in _opcode.hasname:
                arg = self.names[arg]
            elif instr.opcode in _opcode.hasfree:
                if arg < ncells:
                    name = self.cellvars[arg]
                    arg = CellVar(name)
                else:
                    name = self.freevars[arg - ncells]
                    arg = FreeVar(name)
            elif instr.opcode in _opcode.hascompare:
                arg = Compare(arg)
            if jump_target is None:
                instr = Instr(instr.name, arg, lineno=lineno, offset=instr.offset)
            else:
                instr_index = len(instructions)
            instructions.append(instr)
            offset += size // 2 if OFFSET_AS_INSTRUCTION else size
            if jump_target is not None:
                jumps.append((instr_index, jump_target))
        for index, jump_target in jumps:
            instr = instructions[index]
            label = labels[jump_target]
            instructions[index] = Instr(instr.name, label, lineno=instr.lineno, offset=instr.offset)
        bytecode = _bytecode.Bytecode()
        bytecode._copy_attr_from(self)
        nargs = bytecode.argcount + bytecode.kwonlyargcount
        if sys.version_info > (3, 8):
            nargs += bytecode.posonlyargcount
        if bytecode.flags & inspect.CO_VARARGS:
            nargs += 1
        if bytecode.flags & inspect.CO_VARKEYWORDS:
            nargs += 1
        bytecode.argnames = self.varnames[:nargs]
        _set_docstring(bytecode, self.consts)
        bytecode.extend(instructions)
        return bytecode