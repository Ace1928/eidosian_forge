from nltk.parse.api import ParserI
from nltk.tree import Tree
from NLTK's downloader. More unified parsing models can be obtained with
def _scored_parse_to_nltk_tree(scored_parse):
    return Tree.fromstring(str(scored_parse.ptb_parse))