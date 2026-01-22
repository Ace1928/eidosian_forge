import re
import difflib
from collections import namedtuple
import logging
from parso.utils import split_lines
from parso.python.parser import Parser
from parso.python.tree import EndMarker
from parso.python.tokenize import PythonToken, BOM_UTF8_STRING
from parso.python.token import PythonTokenTypes
class DiffParser:
    """
    An advanced form of parsing a file faster. Unfortunately comes with huge
    side effects. It changes the given module.
    """

    def __init__(self, pgen_grammar, tokenizer, module):
        self._pgen_grammar = pgen_grammar
        self._tokenizer = tokenizer
        self._module = module

    def _reset(self):
        self._copy_count = 0
        self._parser_count = 0
        self._nodes_tree = _NodesTree(self._module)

    def update(self, old_lines, new_lines):
        """
        The algorithm works as follows:

        Equal:
            - Assure that the start is a newline, otherwise parse until we get
              one.
            - Copy from parsed_until_line + 1 to max(i2 + 1)
            - Make sure that the indentation is correct (e.g. add DEDENT)
            - Add old and change positions
        Insert:
            - Parse from parsed_until_line + 1 to min(j2 + 1), hopefully not
              much more.

        Returns the new module node.
        """
        LOG.debug('diff parser start')
        self._module._used_names = None
        self._parser_lines_new = new_lines
        self._reset()
        line_length = len(new_lines)
        sm = difflib.SequenceMatcher(None, old_lines, self._parser_lines_new)
        opcodes = sm.get_opcodes()
        LOG.debug('line_lengths old: %s; new: %s' % (len(old_lines), line_length))
        for operation, i1, i2, j1, j2 in opcodes:
            LOG.debug('-> code[%s] old[%s:%s] new[%s:%s]', operation, i1 + 1, i2, j1 + 1, j2)
            if j2 == line_length and new_lines[-1] == '':
                j2 -= 1
            if operation == 'equal':
                line_offset = j1 - i1
                self._copy_from_old_parser(line_offset, i1 + 1, i2, j2)
            elif operation == 'replace':
                self._parse(until_line=j2)
            elif operation == 'insert':
                self._parse(until_line=j2)
            else:
                assert operation == 'delete'
        self._nodes_tree.close()
        if DEBUG_DIFF_PARSER:
            try:
                code = ''.join(new_lines)
                assert self._module.get_code() == code
                _assert_valid_graph(self._module)
                without_diff_parser_module = Parser(self._pgen_grammar, error_recovery=True).parse(self._tokenizer(new_lines))
                _assert_nodes_are_equal(self._module, without_diff_parser_module)
            except AssertionError:
                print(_get_debug_error_message(self._module, old_lines, new_lines))
                raise
        last_pos = self._module.end_pos[0]
        if last_pos != line_length:
            raise Exception('(%s != %s) ' % (last_pos, line_length) + _get_debug_error_message(self._module, old_lines, new_lines))
        LOG.debug('diff parser end')
        return self._module

    def _enabled_debugging(self, old_lines, lines_new):
        if self._module.get_code() != ''.join(lines_new):
            LOG.warning('parser issue:\n%s\n%s', ''.join(old_lines), ''.join(lines_new))

    def _copy_from_old_parser(self, line_offset, start_line_old, until_line_old, until_line_new):
        last_until_line = -1
        while until_line_new > self._nodes_tree.parsed_until_line:
            parsed_until_line_old = self._nodes_tree.parsed_until_line - line_offset
            line_stmt = self._get_old_line_stmt(parsed_until_line_old + 1)
            if line_stmt is None:
                self._parse(self._nodes_tree.parsed_until_line + 1)
            else:
                p_children = line_stmt.parent.children
                index = p_children.index(line_stmt)
                if start_line_old == 1 and p_children[0].get_first_leaf().prefix.startswith(BOM_UTF8_STRING):
                    copied_nodes = []
                else:
                    from_ = self._nodes_tree.parsed_until_line + 1
                    copied_nodes = self._nodes_tree.copy_nodes(p_children[index:], until_line_old, line_offset)
                if copied_nodes:
                    self._copy_count += 1
                    to = self._nodes_tree.parsed_until_line
                    LOG.debug('copy old[%s:%s] new[%s:%s]', copied_nodes[0].start_pos[0], copied_nodes[-1].end_pos[0] - 1, from_, to)
                else:
                    self._parse(self._nodes_tree.parsed_until_line + 1)
            assert last_until_line != self._nodes_tree.parsed_until_line, last_until_line
            last_until_line = self._nodes_tree.parsed_until_line

    def _get_old_line_stmt(self, old_line):
        leaf = self._module.get_leaf_for_position((old_line, 0), include_prefixes=True)
        if _ends_with_newline(leaf):
            leaf = leaf.get_next_leaf()
        if leaf.get_start_pos_of_prefix()[0] == old_line:
            node = leaf
            while node.parent.type not in ('file_input', 'suite'):
                node = node.parent
            if node.start_pos[0] >= old_line:
                return node
        return None

    def _parse(self, until_line):
        """
        Parses at least until the given line, but might just parse more until a
        valid state is reached.
        """
        last_until_line = 0
        while until_line > self._nodes_tree.parsed_until_line:
            node = self._try_parse_part(until_line)
            nodes = node.children
            self._nodes_tree.add_parsed_nodes(nodes, self._keyword_token_indents)
            if self._replace_tos_indent is not None:
                self._nodes_tree.indents[-1] = self._replace_tos_indent
            LOG.debug('parse_part from %s to %s (to %s in part parser)', nodes[0].get_start_pos_of_prefix()[0], self._nodes_tree.parsed_until_line, node.end_pos[0] - 1)
            assert last_until_line != self._nodes_tree.parsed_until_line, last_until_line
            last_until_line = self._nodes_tree.parsed_until_line

    def _try_parse_part(self, until_line):
        """
        Sets up a normal parser that uses a spezialized tokenizer to only parse
        until a certain position (or a bit longer if the statement hasn't
        ended.
        """
        self._parser_count += 1
        parsed_until_line = self._nodes_tree.parsed_until_line
        lines_after = self._parser_lines_new[parsed_until_line:]
        tokens = self._diff_tokenize(lines_after, until_line, line_offset=parsed_until_line)
        self._active_parser = Parser(self._pgen_grammar, error_recovery=True)
        return self._active_parser.parse(tokens=tokens)

    def _diff_tokenize(self, lines, until_line, line_offset=0):
        was_newline = False
        indents = self._nodes_tree.indents
        initial_indentation_count = len(indents)
        tokens = self._tokenizer(lines, start_pos=(line_offset + 1, 0), indents=indents, is_first_token=line_offset == 0)
        stack = self._active_parser.stack
        self._replace_tos_indent = None
        self._keyword_token_indents = {}
        for token in tokens:
            typ = token.type
            if typ == DEDENT:
                if len(indents) < initial_indentation_count:
                    while True:
                        typ, string, start_pos, prefix = token = next(tokens)
                        if typ in (DEDENT, ERROR_DEDENT):
                            if typ == ERROR_DEDENT:
                                self._replace_tos_indent = start_pos[1] + 1
                                pass
                        else:
                            break
                    if '\n' in prefix or '\r' in prefix:
                        prefix = re.sub('[^\\n\\r]+\\Z', '', prefix)
                    else:
                        assert start_pos[1] >= len(prefix), repr(prefix)
                        if start_pos[1] - len(prefix) == 0:
                            prefix = ''
                    yield PythonToken(ENDMARKER, '', start_pos, prefix)
                    break
            elif typ == NEWLINE and token.start_pos[0] >= until_line:
                was_newline = True
            elif was_newline:
                was_newline = False
                if len(indents) == initial_indentation_count:
                    if _suite_or_file_input_is_valid(self._pgen_grammar, stack):
                        yield PythonToken(ENDMARKER, '', token.start_pos, '')
                        break
            if typ == NAME and token.string in ('class', 'def'):
                self._keyword_token_indents[token.start_pos] = list(indents)
            yield token