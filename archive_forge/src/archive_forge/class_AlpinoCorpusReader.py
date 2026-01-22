import sys
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tag import map_tag
from nltk.tree import Tree
class AlpinoCorpusReader(BracketParseCorpusReader):
    """
    Reader for the Alpino Dutch Treebank.
    This corpus has a lexical breakdown structure embedded, as read by `_parse`
    Unfortunately this puts punctuation and some other words out of the sentence
    order in the xml element tree. This is no good for `tag_` and `word_`
    `_tag` and `_word` will be overridden to use a non-default new parameter 'ordered'
    to the overridden _normalize function. The _parse function can then remain
    untouched.
    """

    def __init__(self, root, encoding='ISO-8859-1', tagset=None):
        BracketParseCorpusReader.__init__(self, root, 'alpino\\.xml', detect_blocks='blankline', encoding=encoding, tagset=tagset)

    def _normalize(self, t, ordered=False):
        """Normalize the xml sentence element in t.
        The sentence elements <alpino_ds>, although embedded in a few overall
        xml elements, are separated by blank lines. That's how the reader can
        deliver them one at a time.
        Each sentence has a few category subnodes that are of no use to us.
        The remaining word nodes may or may not appear in the proper order.
        Each word node has attributes, among which:
        - begin : the position of the word in the sentence
        - pos   : Part of Speech: the Tag
        - word  : the actual word
        The return value is a string with all xml elementes replaced by
        clauses: either a cat clause with nested clauses, or a word clause.
        The order of the bracket clauses closely follows the xml.
        If ordered == True, the word clauses include an order sequence number.
        If ordered == False, the word clauses only have pos and word parts.
        """
        if t[:10] != '<alpino_ds':
            return ''
        t = re.sub('  <node .*? cat="(\\w+)".*>', '(\\1', t)
        if ordered:
            t = re.sub('  <node. *?begin="(\\d+)".*? pos="(\\w+)".*? word="([^"]+)".*?/>', '(\\1 \\2 \\3)', t)
        else:
            t = re.sub('  <node .*?pos="(\\w+)".*? word="([^"]+)".*?/>', '(\\1 \\2)', t)
        t = re.sub('  </node>', ')', t)
        t = re.sub('<sentence>.*</sentence>', '', t)
        t = re.sub('</?alpino_ds.*>', '', t)
        return t

    def _tag(self, t, tagset=None):
        tagged_sent = [(int(o), w, p) for o, p, w in SORTTAGWRD.findall(self._normalize(t, ordered=True))]
        tagged_sent.sort()
        if tagset and tagset != self._tagset:
            tagged_sent = [(w, map_tag(self._tagset, tagset, p)) for o, w, p in tagged_sent]
        else:
            tagged_sent = [(w, p) for o, w, p in tagged_sent]
        return tagged_sent

    def _word(self, t):
        """Return a correctly ordered list if words"""
        tagged_sent = self._tag(t)
        return [w for w, p in tagged_sent]