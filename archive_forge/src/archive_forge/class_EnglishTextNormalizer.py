import re
import unicodedata
from fractions import Fraction
from typing import Iterator, List, Match, Optional, Union
import regex
class EnglishTextNormalizer:

    def __init__(self, english_spelling_mapping):
        self.ignore_patterns = '\\b(hmm|mm|mhm|mmm|uh|um)\\b'
        self.replacers = {"\\bwon't\\b": 'will not', "\\bcan't\\b": 'can not', "\\blet's\\b": 'let us', "\\bain't\\b": 'aint', "\\by'all\\b": 'you all', '\\bwanna\\b': 'want to', '\\bgotta\\b': 'got to', '\\bgonna\\b': 'going to', "\\bi'ma\\b": 'i am going to', '\\bimma\\b': 'i am going to', '\\bwoulda\\b': 'would have', '\\bcoulda\\b': 'could have', '\\bshoulda\\b': 'should have', "\\bma'am\\b": 'madam', '\\bmr\\b': 'mister ', '\\bmrs\\b': 'missus ', '\\bst\\b': 'saint ', '\\bdr\\b': 'doctor ', '\\bprof\\b': 'professor ', '\\bcapt\\b': 'captain ', '\\bgov\\b': 'governor ', '\\bald\\b': 'alderman ', '\\bgen\\b': 'general ', '\\bsen\\b': 'senator ', '\\brep\\b': 'representative ', '\\bpres\\b': 'president ', '\\brev\\b': 'reverend ', '\\bhon\\b': 'honorable ', '\\basst\\b': 'assistant ', '\\bassoc\\b': 'associate ', '\\blt\\b': 'lieutenant ', '\\bcol\\b': 'colonel ', '\\bjr\\b': 'junior ', '\\bsr\\b': 'senior ', '\\besq\\b': 'esquire ', "'d been\\b": ' had been', "'s been\\b": ' has been', "'d gone\\b": ' had gone', "'s gone\\b": ' has gone', "'d done\\b": ' had done', "'s got\\b": ' has got', "n't\\b": ' not', "'re\\b": ' are', "'s\\b": ' is', "'d\\b": ' would', "'ll\\b": ' will', "'t\\b": ' not', "'ve\\b": ' have', "'m\\b": ' am'}
        self.standardize_numbers = EnglishNumberNormalizer()
        self.standardize_spellings = EnglishSpellingNormalizer(english_spelling_mapping)

    def __call__(self, s: str):
        s = s.lower()
        s = re.sub('[<\\[][^>\\]]*[>\\]]', '', s)
        s = re.sub('\\(([^)]+?)\\)', '', s)
        s = re.sub(self.ignore_patterns, '', s)
        s = re.sub("\\s+'", "'", s)
        for pattern, replacement in self.replacers.items():
            s = re.sub(pattern, replacement, s)
        s = re.sub('(\\d),(\\d)', '\\1\\2', s)
        s = re.sub('\\.([^0-9]|$)', ' \\1', s)
        s = remove_symbols_and_diacritics(s, keep='.%$¢€£')
        s = self.standardize_numbers(s)
        s = self.standardize_spellings(s)
        s = re.sub('[.$¢€£]([^0-9])', ' \\1', s)
        s = re.sub('([^0-9])%', '\\1 ', s)
        s = re.sub('\\s+', ' ', s)
        return s