from nltk.parse.api import ParserI
from nltk.tree import Tree
from NLTK's downloader. More unified parsing models can be obtained with
def _ensure_ascii(words):
    try:
        for i, word in enumerate(words):
            word.encode('ascii')
    except UnicodeEncodeError as e:
        raise ValueError(f"Token {i} ({word!r}) is non-ASCII. BLLIP Parser currently doesn't support non-ASCII inputs.") from e