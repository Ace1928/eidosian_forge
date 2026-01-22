import dis
from _pydevd_bundle.pydevd_collect_bytecode_info import iter_instructions
from _pydev_bundle import pydev_log
import sys
import inspect
from io import StringIO
class _PyCodeToSource(object):

    def __init__(self, co, memo=None):
        if memo is None:
            memo = {}
        self.memo = memo
        self.co = co
        self.instructions = list(iter_instructions(co))
        self.stack = _Stack()
        self.writer = _Writer()

    def _process_next(self, i_line):
        instruction = self.instructions.pop(0)
        handler_class = _op_name_to_handler.get(instruction.opname)
        if handler_class is not None:
            s = handler_class(i_line, instruction, self.stack, self.writer, self)
            if DEBUG:
                print(s)
        elif DEBUG:
            print('UNHANDLED', instruction)

    def build_line_to_contents(self):
        co = self.co
        op_offset_to_line = dict(dis.findlinestarts(co))
        curr_line_index = 0
        instructions = self.instructions
        while instructions:
            instruction = instructions[0]
            new_line_index = op_offset_to_line.get(instruction.offset)
            if new_line_index is not None:
                if new_line_index is not None:
                    curr_line_index = new_line_index
            self._process_next(curr_line_index)
        return self.writer.line_to_contents

    def merge_code(self, code):
        if DEBUG:
            print('merge code ----')
        line_to_contents = _PyCodeToSource(code, self.memo).build_line_to_contents()
        lines = []
        for line, contents in sorted(line_to_contents.items()):
            lines.append(line)
            self.writer.get_line(line).extend(contents)
        if DEBUG:
            print('end merge code ----')
        return lines

    def disassemble(self):
        show_lines = False
        line_to_contents = self.build_line_to_contents()
        stream = StringIO()
        last_line = 0
        indent = ''
        previous_line_tokens = set()
        for i_line, contents in sorted(line_to_contents.items()):
            while last_line < i_line - 1:
                if show_lines:
                    stream.write(u'%s.\n' % (last_line + 1,))
                else:
                    stream.write(u'\n')
                last_line += 1
            line_contents = []
            dedents_found = 0
            for part in contents:
                if part is INDENT_MARKER:
                    if DEBUG:
                        print('found indent', i_line)
                    indent += '    '
                    continue
                if part is DEDENT_MARKER:
                    if DEBUG:
                        print('found dedent', i_line)
                    dedents_found += 1
                    continue
                line_contents.append(part)
            s = indent + _compose_line_contents(line_contents, previous_line_tokens)
            if show_lines:
                stream.write(u'%s. %s\n' % (i_line, s))
            else:
                stream.write(u'%s\n' % s)
            if dedents_found:
                indent = indent[:-(4 * dedents_found)]
            last_line = i_line
        return stream.getvalue()