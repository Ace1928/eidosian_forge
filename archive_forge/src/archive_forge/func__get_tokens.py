from __future__ import unicode_literals
from prompt_toolkit.document import Document
from prompt_toolkit.layout.lexers import Lexer
from prompt_toolkit.layout.utils import split_lines
from prompt_toolkit.token import Token
from .compiler import _CompiledGrammar
from six.moves import range
def _get_tokens(self, cli, text):
    m = self.compiled_grammar.match_prefix(text)
    if m:
        characters = [[self.default_token, c] for c in text]
        for v in m.variables():
            lexer = self.lexers.get(v.varname)
            if lexer:
                document = Document(text[v.start:v.stop])
                lexer_tokens_for_line = lexer.lex_document(cli, document)
                lexer_tokens = []
                for i in range(len(document.lines)):
                    lexer_tokens.extend(lexer_tokens_for_line(i))
                    lexer_tokens.append((Token, '\n'))
                if lexer_tokens:
                    lexer_tokens.pop()
                i = v.start
                for t, s in lexer_tokens:
                    for c in s:
                        if characters[i][0] == self.default_token:
                            characters[i][0] = t
                        i += 1
        trailing_input = m.trailing_input()
        if trailing_input:
            for i in range(trailing_input.start, trailing_input.stop):
                characters[i][0] = Token.TrailingInput
        return characters
    else:
        return [(Token, text)]