import codecs
import dataclasses
import unicodedata
from typing import Optional, List, Union, Any, Iterator, Tuple, Type, Dict
from latexcodec import lexer
from codecs import CodecInfo
class LatexUnicodeTable:
    """Tabulates a translation between LaTeX and unicode."""

    def __init__(self, lexer_):
        self.lexer: lexer.LatexIncrementalLexer = lexer_
        self.unicode_map: Dict[Tuple[lexer.Token, ...], str] = {}
        self.max_length: int = 0
        self.latex_map: Dict[str, Tuple[str, Tuple[lexer.Token, ...]]] = {}
        self.register_all()

    def register_all(self):
        """Register all symbols and their LaTeX equivalents
        (called by constructor).
        """
        self.register(UnicodeLatexTranslation(unicode='\n\n', latex=' \\par', encode=False, decode=True, text_mode=True, math_mode=False))
        self.register(UnicodeLatexTranslation(unicode='\n\n', latex='\\par', encode=False, decode=True, text_mode=True, math_mode=False))
        for trans in load_unicode_latex_table():
            self.register(trans)

    def register(self, trans: UnicodeLatexTranslation):
        """Register a correspondence between *unicode_text* and *latex_text*.

        :param UnicodeLatexTranslation trans: Description of translation.
        """
        if trans.math_mode and (not trans.text_mode):
            self.register(UnicodeLatexTranslation(unicode=trans.unicode, latex='$' + trans.latex + '$', text_mode=True, math_mode=False, decode=trans.decode, encode=trans.encode))
            self.register(UnicodeLatexTranslation(unicode=trans.unicode, latex='\\(' + trans.latex + '\\)', text_mode=True, math_mode=False, decode=trans.decode, encode=trans.encode))
            return
        self.lexer.reset()
        self.lexer.state = 'M'
        tokens = tuple(self.lexer.get_tokens(trans.latex, final=True))
        if trans.decode:
            if tokens not in self.unicode_map:
                self.max_length = max(self.max_length, len(tokens))
                self.unicode_map[tokens] = trans.unicode
            if len(tokens) == 2 and tokens[0].name.startswith('control') and (tokens[1].name == 'chars'):
                self.register(UnicodeLatexTranslation(unicode=f'{{{trans.unicode}}}', latex=f'{tokens[0].text}{{{tokens[1].text}}}', decode=True, encode=False, math_mode=trans.math_mode, text_mode=trans.text_mode))
            if len(tokens) == 4 and tokens[0].text in {'$', '\\('} and tokens[1].name.startswith('control') and (tokens[2].name == 'chars') and (tokens[3].text in {'$', '\\)'}):
                self.register(UnicodeLatexTranslation(unicode=f'{trans.unicode}', latex=f'{tokens[0].text}{tokens[1].text}{{{tokens[2].text}}}{tokens[3].text}', decode=True, encode=False, math_mode=trans.math_mode, text_mode=trans.text_mode))
        if trans.encode and trans.unicode not in self.latex_map:
            assert len(trans.unicode) == 1
            self.latex_map[trans.unicode] = (trans.latex, tokens)