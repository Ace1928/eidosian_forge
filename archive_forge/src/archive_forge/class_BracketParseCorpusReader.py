import sys
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tag import map_tag
from nltk.tree import Tree
class BracketParseCorpusReader(SyntaxCorpusReader):
    """
    Reader for corpora that consist of parenthesis-delineated parse trees,
    like those found in the "combined" section of the Penn Treebank,
    e.g. "(S (NP (DT the) (JJ little) (NN dog)) (VP (VBD barked)))".

    """

    def __init__(self, root, fileids, comment_char=None, detect_blocks='unindented_paren', encoding='utf8', tagset=None):
        """
        :param root: The root directory for this corpus.
        :param fileids: A list or regexp specifying the fileids in this corpus.
        :param comment_char: The character which can appear at the start of
            a line to indicate that the rest of the line is a comment.
        :param detect_blocks: The method that is used to find blocks
            in the corpus; can be 'unindented_paren' (every unindented
            parenthesis starts a new parse) or 'sexpr' (brackets are
            matched).
        :param tagset: The name of the tagset used by this corpus, to be used
            for normalizing or converting the POS tags returned by the
            ``tagged_...()`` methods.
        """
        SyntaxCorpusReader.__init__(self, root, fileids, encoding)
        self._comment_char = comment_char
        self._detect_blocks = detect_blocks
        self._tagset = tagset

    def _read_block(self, stream):
        if self._detect_blocks == 'sexpr':
            return read_sexpr_block(stream, comment_char=self._comment_char)
        elif self._detect_blocks == 'blankline':
            return read_blankline_block(stream)
        elif self._detect_blocks == 'unindented_paren':
            toks = read_regexp_block(stream, start_re='^\\(')
            if self._comment_char:
                toks = [re.sub('(?m)^%s.*' % re.escape(self._comment_char), '', tok) for tok in toks]
            return toks
        else:
            assert 0, 'bad block type'

    def _normalize(self, t):
        t = re.sub('\\((.)\\)', '(\\1 \\1)', t)
        t = re.sub('\\(([^\\s()]+) ([^\\s()]+) [^\\s()]+\\)', '(\\1 \\2)', t)
        return t

    def _parse(self, t):
        try:
            tree = Tree.fromstring(self._normalize(t))
            if tree.label() == '' and len(tree) == 1:
                return tree[0]
            else:
                return tree
        except ValueError as e:
            sys.stderr.write('Bad tree detected; trying to recover...\n')
            if e.args == ('mismatched parens',):
                for n in range(1, 5):
                    try:
                        v = Tree(self._normalize(t + ')' * n))
                        sys.stderr.write('  Recovered by adding %d close paren(s)\n' % n)
                        return v
                    except ValueError:
                        pass
            sys.stderr.write('  Recovered by returning a flat parse.\n')
            return Tree('S', self._tag(t))

    def _tag(self, t, tagset=None):
        tagged_sent = [(w, p) for p, w in TAGWORD.findall(self._normalize(t))]
        if tagset and tagset != self._tagset:
            tagged_sent = [(w, map_tag(self._tagset, tagset, p)) for w, p in tagged_sent]
        return tagged_sent

    def _word(self, t):
        return WORD.findall(self._normalize(t))