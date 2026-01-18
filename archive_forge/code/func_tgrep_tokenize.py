import functools
import re
import nltk.tree
def tgrep_tokenize(tgrep_string):
    """
    Tokenizes a TGrep search string into separate tokens.
    """
    parser = _build_tgrep_parser(False)
    if isinstance(tgrep_string, bytes):
        tgrep_string = tgrep_string.decode()
    return list(parser.parseString(tgrep_string))