from re import finditer
from xml.sax.saxutils import escape, unescape
class CJKChars:
    """
    An object that enumerates the code points of the CJK characters as listed on
    https://en.wikipedia.org/wiki/Basic_Multilingual_Plane#Basic_Multilingual_Plane

    This is a Python port of the CJK code point enumerations of Moses tokenizer:
    https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/detokenizer.perl#L309
    """
    Hangul_Jamo = (4352, 4607)
    CJK_Radicals = (11904, 42191)
    Phags_Pa = (43072, 43135)
    Hangul_Syllables = (44032, 55215)
    CJK_Compatibility_Ideographs = (63744, 64255)
    CJK_Compatibility_Forms = (65072, 65103)
    Katakana_Hangul_Halfwidth = (65381, 65500)
    Supplementary_Ideographic_Plane = (131072, 196607)
    ranges = [Hangul_Jamo, CJK_Radicals, Phags_Pa, Hangul_Syllables, CJK_Compatibility_Ideographs, CJK_Compatibility_Forms, Katakana_Hangul_Halfwidth, Supplementary_Ideographic_Plane]