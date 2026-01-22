import ast
from codeop import CommandCompiler, Compile
import re
import sys
import tokenize
from typing import List, Tuple, Optional, Any
import warnings
from IPython.utils import tokenutil
class TransformerManager:
    """Applies various transformations to a cell or code block.

    The key methods for external use are ``transform_cell()``
    and ``check_complete()``.
    """

    def __init__(self):
        self.cleanup_transforms = [leading_empty_lines, leading_indent, classic_prompt, ipython_prompt]
        self.line_transforms = [cell_magic]
        self.token_transformers = [MagicAssign, SystemAssign, EscapedCommand, HelpEnd]

    def do_one_token_transform(self, lines):
        """Find and run the transform earliest in the code.

        Returns (changed, lines).

        This method is called repeatedly until changed is False, indicating
        that all available transformations are complete.

        The tokens following IPython special syntax might not be valid, so
        the transformed code is retokenised every time to identify the next
        piece of special syntax. Hopefully long code cells are mostly valid
        Python, not using lots of IPython special syntax, so this shouldn't be
        a performance issue.
        """
        tokens_by_line = make_tokens_by_line(lines)
        candidates = []
        for transformer_cls in self.token_transformers:
            transformer = transformer_cls.find(tokens_by_line)
            if transformer:
                candidates.append(transformer)
        if not candidates:
            return (False, lines)
        ordered_transformers = sorted(candidates, key=TokenTransformBase.sortby)
        for transformer in ordered_transformers:
            try:
                return (True, transformer.transform(lines))
            except SyntaxError:
                pass
        return (False, lines)

    def do_token_transforms(self, lines):
        for _ in range(TRANSFORM_LOOP_LIMIT):
            changed, lines = self.do_one_token_transform(lines)
            if not changed:
                return lines
        raise RuntimeError('Input transformation still changing after %d iterations. Aborting.' % TRANSFORM_LOOP_LIMIT)

    def transform_cell(self, cell: str) -> str:
        """Transforms a cell of input code"""
        if not cell.endswith('\n'):
            cell += '\n'
        lines = cell.splitlines(keepends=True)
        for transform in self.cleanup_transforms + self.line_transforms:
            lines = transform(lines)
        lines = self.do_token_transforms(lines)
        return ''.join(lines)

    def check_complete(self, cell: str):
        """Return whether a block of code is ready to execute, or should be continued

        Parameters
        ----------
        cell : string
            Python input code, which can be multiline.

        Returns
        -------
        status : str
            One of 'complete', 'incomplete', or 'invalid' if source is not a
            prefix of valid code.
        indent_spaces : int or None
            The number of spaces by which to indent the next line of code. If
            status is not 'incomplete', this is None.
        """
        ends_with_newline = False
        for character in reversed(cell):
            if character == '\n':
                ends_with_newline = True
                break
            elif character.strip():
                break
            else:
                continue
        if not ends_with_newline:
            cell += '\n'
        lines = cell.splitlines(keepends=True)
        if not lines:
            return ('complete', None)
        for line in reversed(lines):
            if not line.strip():
                continue
            elif line.strip('\n').endswith('\\'):
                return ('incomplete', find_last_indent(lines))
            else:
                break
        try:
            for transform in self.cleanup_transforms:
                if not getattr(transform, 'has_side_effects', False):
                    lines = transform(lines)
        except SyntaxError:
            return ('invalid', None)
        if lines[0].startswith('%%'):
            if lines[-1].strip():
                return ('incomplete', find_last_indent(lines))
            else:
                return ('complete', None)
        try:
            for transform in self.line_transforms:
                if not getattr(transform, 'has_side_effects', False):
                    lines = transform(lines)
            lines = self.do_token_transforms(lines)
        except SyntaxError:
            return ('invalid', None)
        tokens_by_line = make_tokens_by_line(lines)
        if len(lines) == 1 and tokens_by_line and has_sunken_brackets(tokens_by_line[0]):
            return ('invalid', None)
        if not tokens_by_line:
            return ('incomplete', find_last_indent(lines))
        if tokens_by_line[-1][-1].type != tokenize.ENDMARKER and tokens_by_line[-1][-1].type != tokenize.ERRORTOKEN:
            return ('incomplete', find_last_indent(lines))
        newline_types = {tokenize.NEWLINE, tokenize.COMMENT, tokenize.ENDMARKER}
        last_token_line = None
        if {t.type for t in tokens_by_line[-1]} in [{tokenize.DEDENT, tokenize.ENDMARKER}, {tokenize.ENDMARKER}] and len(tokens_by_line) > 1:
            last_token_line = tokens_by_line.pop()
        while tokens_by_line[-1] and tokens_by_line[-1][-1].type in newline_types:
            tokens_by_line[-1].pop()
        if not tokens_by_line[-1]:
            return ('incomplete', find_last_indent(lines))
        if tokens_by_line[-1][-1].string == ':':
            ix = 0
            while tokens_by_line[-1][ix].type in {tokenize.INDENT, tokenize.DEDENT}:
                ix += 1
            indent = tokens_by_line[-1][ix].start[1]
            return ('incomplete', indent + 4)
        if tokens_by_line[-1][0].line.endswith('\\'):
            return ('incomplete', None)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', SyntaxWarning)
                res = compile_command(''.join(lines), symbol='exec')
        except (SyntaxError, OverflowError, ValueError, TypeError, MemoryError, SyntaxWarning):
            return ('invalid', None)
        else:
            if res is None:
                return ('incomplete', find_last_indent(lines))
        if last_token_line and last_token_line[0].type == tokenize.DEDENT:
            if ends_with_newline:
                return ('complete', None)
            return ('incomplete', find_last_indent(lines))
        if not lines[-1].strip():
            return ('complete', None)
        return ('complete', None)