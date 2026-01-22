import dis
import inspect
import sys
from collections import namedtuple
from _pydev_bundle import pydev_log
from opcode import (EXTENDED_ARG, HAVE_ARGUMENT, cmp_op, hascompare, hasconst,
from io import StringIO
import ast as ast_module
class _Disassembler(object):

    def __init__(self, co, firstlineno, level=0):
        self.co = co
        self.firstlineno = firstlineno
        self.level = level
        self.instructions = list(iter_instructions(co))
        op_offset_to_line = self.op_offset_to_line = dict(dis.findlinestarts(co))
        line_index = co.co_firstlineno - firstlineno
        for instruction in self.instructions:
            new_line_index = op_offset_to_line.get(instruction.offset)
            if new_line_index is not None:
                line_index = new_line_index - firstlineno
                op_offset_to_line[instruction.offset] = line_index
            else:
                op_offset_to_line[instruction.offset] = line_index
    BIG_LINE_INT = 9999999
    SMALL_LINE_INT = -1

    def min_line(self, *args):
        m = self.BIG_LINE_INT
        for arg in args:
            if isinstance(arg, (list, tuple)):
                m = min(m, self.min_line(*arg))
            elif isinstance(arg, _MsgPart):
                m = min(m, arg.line)
            elif hasattr(arg, 'offset'):
                m = min(m, self.op_offset_to_line[arg.offset])
        return m

    def max_line(self, *args):
        m = self.SMALL_LINE_INT
        for arg in args:
            if isinstance(arg, (list, tuple)):
                m = max(m, self.max_line(*arg))
            elif isinstance(arg, _MsgPart):
                m = max(m, arg.line)
            elif hasattr(arg, 'offset'):
                m = max(m, self.op_offset_to_line[arg.offset])
        return m

    def _lookahead(self):
        """
        This handles and converts some common constructs from bytecode to actual source code.

        It may change the list of instructions.
        """
        msg = self._create_msg_part
        found = []
        fullrepr = None
        for next_instruction in self.instructions:
            if next_instruction.opname in ('LOAD_GLOBAL', 'LOAD_FAST', 'LOAD_CONST', 'LOAD_NAME'):
                found.append(next_instruction)
            else:
                break
        if not found:
            return None
        if next_instruction.opname == 'LOAD_ATTR':
            prev_instruction = found[-1]
            assert self.instructions.pop(len(found)) is next_instruction
            self.instructions[len(found) - 1] = _Instruction(prev_instruction.opname, prev_instruction.opcode, prev_instruction.starts_line, prev_instruction.argval, False, prev_instruction.offset, (msg(prev_instruction), msg(prev_instruction, '.'), msg(next_instruction)))
            return RESTART_FROM_LOOKAHEAD
        if next_instruction.opname in ('CALL_FUNCTION', 'PRECALL'):
            if len(found) == next_instruction.argval + 1:
                force_restart = False
                delta = 0
            else:
                force_restart = True
                if len(found) > next_instruction.argval + 1:
                    delta = len(found) - (next_instruction.argval + 1)
                else:
                    return None
            del_upto = delta + next_instruction.argval + 2
            if next_instruction.opname == 'PRECALL':
                del_upto += 1
            del self.instructions[delta:del_upto]
            found = iter(found[delta:])
            call_func = next(found)
            args = list(found)
            fullrepr = [msg(call_func), msg(call_func, '(')]
            prev = call_func
            for i, arg in enumerate(args):
                if i > 0:
                    fullrepr.append(msg(prev, ', '))
                prev = arg
                fullrepr.append(msg(arg))
            fullrepr.append(msg(prev, ')'))
            if force_restart:
                self.instructions.insert(delta, _Instruction(call_func.opname, call_func.opcode, call_func.starts_line, call_func.argval, False, call_func.offset, tuple(fullrepr)))
                return RESTART_FROM_LOOKAHEAD
        elif next_instruction.opname == 'BUILD_TUPLE':
            if len(found) == next_instruction.argval:
                force_restart = False
                delta = 0
            else:
                force_restart = True
                if len(found) > next_instruction.argval:
                    delta = len(found) - next_instruction.argval
                else:
                    return None
            del self.instructions[delta:delta + next_instruction.argval + 1]
            found = iter(found[delta:])
            args = [instruction for instruction in found]
            if args:
                first_instruction = args[0]
            else:
                first_instruction = next_instruction
            prev = first_instruction
            fullrepr = []
            fullrepr.append(msg(prev, '('))
            for i, arg in enumerate(args):
                if i > 0:
                    fullrepr.append(msg(prev, ', '))
                prev = arg
                fullrepr.append(msg(arg))
            fullrepr.append(msg(prev, ')'))
            if force_restart:
                self.instructions.insert(delta, _Instruction(first_instruction.opname, first_instruction.opcode, first_instruction.starts_line, first_instruction.argval, False, first_instruction.offset, tuple(fullrepr)))
                return RESTART_FROM_LOOKAHEAD
        if fullrepr is not None and self.instructions:
            if self.instructions[0].opname == 'POP_TOP':
                self.instructions.pop(0)
            if self.instructions[0].opname in ('STORE_FAST', 'STORE_NAME'):
                next_instruction = self.instructions.pop(0)
                return (msg(next_instruction), msg(next_instruction, ' = '), fullrepr)
            if self.instructions[0].opname == 'RETURN_VALUE':
                next_instruction = self.instructions.pop(0)
                return (msg(next_instruction, 'return ', line=self.min_line(next_instruction, fullrepr)), fullrepr)
        return fullrepr

    def _decorate_jump_target(self, instruction, instruction_repr):
        if instruction.is_jump_target:
            return ('|', str(instruction.offset), '|', instruction_repr)
        return instruction_repr

    def _create_msg_part(self, instruction, tok=None, line=None):
        dec = self._decorate_jump_target
        if line is None or line in (self.BIG_LINE_INT, self.SMALL_LINE_INT):
            line = self.op_offset_to_line[instruction.offset]
        argrepr = instruction.argrepr
        if isinstance(argrepr, str) and argrepr.startswith('NULL + '):
            argrepr = argrepr[7:]
        return _MsgPart(line, tok if tok is not None else dec(instruction, argrepr))

    def _next_instruction_to_str(self, line_to_contents):
        if self.instructions:
            ret = self._lookahead()
            if ret:
                return ret
        msg = self._create_msg_part
        instruction = self.instructions.pop(0)
        if instruction.opname in 'RESUME':
            return None
        if instruction.opname in ('LOAD_GLOBAL', 'LOAD_FAST', 'LOAD_CONST', 'LOAD_NAME'):
            next_instruction = self.instructions[0]
            if next_instruction.opname in ('STORE_FAST', 'STORE_NAME'):
                self.instructions.pop(0)
                return (msg(next_instruction), msg(next_instruction, ' = '), msg(instruction))
            if next_instruction.opname == 'RETURN_VALUE':
                self.instructions.pop(0)
                return (msg(instruction, 'return ', line=self.min_line(instruction)), msg(instruction))
            if next_instruction.opname == 'RAISE_VARARGS' and next_instruction.argval == 1:
                self.instructions.pop(0)
                return (msg(instruction, 'raise ', line=self.min_line(instruction)), msg(instruction))
        if instruction.opname == 'LOAD_CONST':
            if inspect.iscode(instruction.argval):
                code_line_to_contents = _Disassembler(instruction.argval, self.firstlineno, self.level + 1).build_line_to_contents()
                for contents in code_line_to_contents.values():
                    contents.insert(0, '    ')
                for line, contents in code_line_to_contents.items():
                    line_to_contents.setdefault(line, []).extend(contents)
                return msg(instruction, 'LOAD_CONST(code)')
        if instruction.opname == 'RAISE_VARARGS':
            if instruction.argval == 0:
                return msg(instruction, 'raise')
        if instruction.opname == 'SETUP_FINALLY':
            return msg(instruction, ('try(', instruction.argrepr, '):'))
        if instruction.argrepr:
            return msg(instruction, (instruction.opname, '(', instruction.argrepr, ')'))
        if instruction.argval:
            return msg(instruction, '%s{%s}' % (instruction.opname, instruction.argval))
        return msg(instruction, instruction.opname)

    def build_line_to_contents(self):
        line_to_contents = {}
        instructions = self.instructions
        while instructions:
            s = self._next_instruction_to_str(line_to_contents)
            if s is RESTART_FROM_LOOKAHEAD:
                continue
            if s is None:
                continue
            _MsgPart.add_to_line_to_contents(s, line_to_contents)
            m = self.max_line(s)
            if m != self.SMALL_LINE_INT:
                line_to_contents.setdefault(m, []).append(SEPARATOR)
        return line_to_contents

    def disassemble(self):
        line_to_contents = self.build_line_to_contents()
        stream = StringIO()
        last_line = 0
        show_lines = False
        for line, contents in sorted(line_to_contents.items()):
            while last_line < line - 1:
                if show_lines:
                    stream.write('%s.\n' % (last_line + 1,))
                else:
                    stream.write('\n')
                last_line += 1
            if show_lines:
                stream.write('%s. ' % (line,))
            for i, content in enumerate(contents):
                if content == SEPARATOR:
                    if i != len(contents) - 1:
                        stream.write(', ')
                else:
                    stream.write(content)
            stream.write('\n')
            last_line = line
        return stream.getvalue()