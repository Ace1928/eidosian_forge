import re
from pygments.lexer import Lexer, RegexLexer, bygroups, words, do_insertions, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class ElixirLexer(RegexLexer):
    """
    For the `Elixir language <http://elixir-lang.org>`_.

    .. versionadded:: 1.5
    """
    name = 'Elixir'
    aliases = ['elixir', 'ex', 'exs']
    filenames = ['*.ex', '*.exs']
    mimetypes = ['text/x-elixir']
    KEYWORD = ('fn', 'do', 'end', 'after', 'else', 'rescue', 'catch')
    KEYWORD_OPERATOR = ('not', 'and', 'or', 'when', 'in')
    BUILTIN = ('case', 'cond', 'for', 'if', 'unless', 'try', 'receive', 'raise', 'quote', 'unquote', 'unquote_splicing', 'throw', 'super')
    BUILTIN_DECLARATION = ('def', 'defp', 'defmodule', 'defprotocol', 'defmacro', 'defmacrop', 'defdelegate', 'defexception', 'defstruct', 'defimpl', 'defcallback')
    BUILTIN_NAMESPACE = ('import', 'require', 'use', 'alias')
    CONSTANT = ('nil', 'true', 'false')
    PSEUDO_VAR = ('_', '__MODULE__', '__DIR__', '__ENV__', '__CALLER__')
    OPERATORS3 = ('<<<', '>>>', '|||', '&&&', '^^^', '~~~', '===', '!==', '~>>', '<~>', '|~>', '<|>')
    OPERATORS2 = ('==', '!=', '<=', '>=', '&&', '||', '<>', '++', '--', '|>', '=~', '->', '<-', '|', '.', '=', '~>', '<~')
    OPERATORS1 = ('<', '>', '+', '-', '*', '/', '!', '^', '&')
    PUNCTUATION = ('\\\\', '<<', '>>', '=>', '(', ')', ':', ';', ',', '[', ']')

    def get_tokens_unprocessed(self, text):
        for index, token, value in RegexLexer.get_tokens_unprocessed(self, text):
            if token is Name:
                if value in self.KEYWORD:
                    yield (index, Keyword, value)
                elif value in self.KEYWORD_OPERATOR:
                    yield (index, Operator.Word, value)
                elif value in self.BUILTIN:
                    yield (index, Keyword, value)
                elif value in self.BUILTIN_DECLARATION:
                    yield (index, Keyword.Declaration, value)
                elif value in self.BUILTIN_NAMESPACE:
                    yield (index, Keyword.Namespace, value)
                elif value in self.CONSTANT:
                    yield (index, Name.Constant, value)
                elif value in self.PSEUDO_VAR:
                    yield (index, Name.Builtin.Pseudo, value)
                else:
                    yield (index, token, value)
            else:
                yield (index, token, value)

    def gen_elixir_sigil_rules():
        terminators = [('\\{', '\\}', 'cb'), ('\\[', '\\]', 'sb'), ('\\(', '\\)', 'pa'), ('<', '>', 'ab'), ('/', '/', 'slas'), ('\\|', '\\|', 'pipe'), ('"', '"', 'quot'), ("'", "'", 'apos')]
        triquotes = [('"""', 'triquot'), ("'''", 'triapos')]
        token = String.Other
        states = {'sigils': []}
        for term, name in triquotes:
            states['sigils'] += [('(~[a-z])(%s)' % (term,), bygroups(token, String.Heredoc), (name + '-end', name + '-intp')), ('(~[A-Z])(%s)' % (term,), bygroups(token, String.Heredoc), (name + '-end', name + '-no-intp'))]
            states[name + '-end'] = [('[a-zA-Z]+', token, '#pop'), default('#pop')]
            states[name + '-intp'] = [('^\\s*' + term, String.Heredoc, '#pop'), include('heredoc_interpol')]
            states[name + '-no-intp'] = [('^\\s*' + term, String.Heredoc, '#pop'), include('heredoc_no_interpol')]
        for lterm, rterm, name in terminators:
            states['sigils'] += [('~[a-z]' + lterm, token, name + '-intp'), ('~[A-Z]' + lterm, token, name + '-no-intp')]
            states[name + '-intp'] = gen_elixir_sigstr_rules(rterm, token)
            states[name + '-no-intp'] = gen_elixir_sigstr_rules(rterm, token, interpol=False)
        return states
    op3_re = '|'.join((re.escape(s) for s in OPERATORS3))
    op2_re = '|'.join((re.escape(s) for s in OPERATORS2))
    op1_re = '|'.join((re.escape(s) for s in OPERATORS1))
    ops_re = '(?:%s|%s|%s)' % (op3_re, op2_re, op1_re)
    punctuation_re = '|'.join((re.escape(s) for s in PUNCTUATION))
    alnum = '\\w'
    name_re = '(?:\\.\\.\\.|[a-z_]%s*[!?]?)' % alnum
    modname_re = '[A-Z]%(alnum)s*(?:\\.[A-Z]%(alnum)s*)*' % {'alnum': alnum}
    complex_name_re = '(?:%s|%s|%s)' % (name_re, modname_re, ops_re)
    special_atom_re = '(?:\\.\\.\\.|<<>>|%\\{\\}|%|\\{\\})'
    long_hex_char_re = '(\\\\x\\{)([\\da-fA-F]+)(\\})'
    hex_char_re = '(\\\\x[\\da-fA-F]{1,2})'
    escape_char_re = '(\\\\[abdefnrstv])'
    tokens = {'root': [('\\s+', Text), ('#.*$', Comment.Single), ('(\\?)' + long_hex_char_re, bygroups(String.Char, String.Escape, Number.Hex, String.Escape)), ('(\\?)' + hex_char_re, bygroups(String.Char, String.Escape)), ('(\\?)' + escape_char_re, bygroups(String.Char, String.Escape)), ('\\?\\\\?.', String.Char), (':::', String.Symbol), ('::', Operator), (':' + special_atom_re, String.Symbol), (':' + complex_name_re, String.Symbol), (':"', String.Symbol, 'string_double_atom'), (":'", String.Symbol, 'string_single_atom'), ('(%s|%s)(:)(?=\\s|\\n)' % (special_atom_re, complex_name_re), bygroups(String.Symbol, Punctuation)), ('@' + name_re, Name.Attribute), (name_re, Name), ('(%%?)(%s)' % (modname_re,), bygroups(Punctuation, Name.Class)), (op3_re, Operator), (op2_re, Operator), (punctuation_re, Punctuation), ('&\\d', Name.Entity), (op1_re, Operator), ('0b[01]+', Number.Bin), ('0o[0-7]+', Number.Oct), ('0x[\\da-fA-F]+', Number.Hex), ('\\d(_?\\d)*\\.\\d(_?\\d)*([eE][-+]?\\d(_?\\d)*)?', Number.Float), ('\\d(_?\\d)*', Number.Integer), ('"""\\s*', String.Heredoc, 'heredoc_double'), ("'''\\s*$", String.Heredoc, 'heredoc_single'), ('"', String.Double, 'string_double'), ("'", String.Single, 'string_single'), include('sigils'), ('%\\{', Punctuation, 'map_key'), ('\\{', Punctuation, 'tuple')], 'heredoc_double': [('^\\s*"""', String.Heredoc, '#pop'), include('heredoc_interpol')], 'heredoc_single': [("^\\s*'''", String.Heredoc, '#pop'), include('heredoc_interpol')], 'heredoc_interpol': [('[^#\\\\\\n]+', String.Heredoc), include('escapes'), ('\\\\.', String.Heredoc), ('\\n+', String.Heredoc), include('interpol')], 'heredoc_no_interpol': [('[^\\\\\\n]+', String.Heredoc), ('\\\\.', String.Heredoc), ('\\n+', String.Heredoc)], 'escapes': [(long_hex_char_re, bygroups(String.Escape, Number.Hex, String.Escape)), (hex_char_re, String.Escape), (escape_char_re, String.Escape)], 'interpol': [('#\\{', String.Interpol, 'interpol_string')], 'interpol_string': [('\\}', String.Interpol, '#pop'), include('root')], 'map_key': [include('root'), (':', Punctuation, 'map_val'), ('=>', Punctuation, 'map_val'), ('\\}', Punctuation, '#pop')], 'map_val': [include('root'), (',', Punctuation, '#pop'), ('(?=\\})', Punctuation, '#pop')], 'tuple': [include('root'), ('\\}', Punctuation, '#pop')]}
    tokens.update(gen_elixir_string_rules('double', '"', String.Double))
    tokens.update(gen_elixir_string_rules('single', "'", String.Single))
    tokens.update(gen_elixir_string_rules('double_atom', '"', String.Symbol))
    tokens.update(gen_elixir_string_rules('single_atom', "'", String.Symbol))
    tokens.update(gen_elixir_sigil_rules())