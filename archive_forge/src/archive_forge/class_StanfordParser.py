import os
import tempfile
import warnings
from subprocess import PIPE
from nltk.internals import (
from nltk.parse.api import ParserI
from nltk.parse.dependencygraph import DependencyGraph
from nltk.tree import Tree
class StanfordParser(GenericStanfordParser):
    """
    >>> parser=StanfordParser(
    ...     model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz"
    ... ) # doctest: +SKIP

    >>> list(parser.raw_parse("the quick brown fox jumps over the lazy dog")) # doctest: +NORMALIZE_WHITESPACE +SKIP
    [Tree('ROOT', [Tree('NP', [Tree('NP', [Tree('DT', ['the']), Tree('JJ', ['quick']), Tree('JJ', ['brown']),
    Tree('NN', ['fox'])]), Tree('NP', [Tree('NP', [Tree('NNS', ['jumps'])]), Tree('PP', [Tree('IN', ['over']),
    Tree('NP', [Tree('DT', ['the']), Tree('JJ', ['lazy']), Tree('NN', ['dog'])])])])])])]

    >>> sum([list(dep_graphs) for dep_graphs in parser.raw_parse_sents((
    ...     "the quick brown fox jumps over the lazy dog",
    ...     "the quick grey wolf jumps over the lazy fox"
    ... ))], []) # doctest: +NORMALIZE_WHITESPACE +SKIP
    [Tree('ROOT', [Tree('NP', [Tree('NP', [Tree('DT', ['the']), Tree('JJ', ['quick']), Tree('JJ', ['brown']),
    Tree('NN', ['fox'])]), Tree('NP', [Tree('NP', [Tree('NNS', ['jumps'])]), Tree('PP', [Tree('IN', ['over']),
    Tree('NP', [Tree('DT', ['the']), Tree('JJ', ['lazy']), Tree('NN', ['dog'])])])])])]), Tree('ROOT', [Tree('NP',
    [Tree('NP', [Tree('DT', ['the']), Tree('JJ', ['quick']), Tree('JJ', ['grey']), Tree('NN', ['wolf'])]), Tree('NP',
    [Tree('NP', [Tree('NNS', ['jumps'])]), Tree('PP', [Tree('IN', ['over']), Tree('NP', [Tree('DT', ['the']),
    Tree('JJ', ['lazy']), Tree('NN', ['fox'])])])])])])]

    >>> sum([list(dep_graphs) for dep_graphs in parser.parse_sents((
    ...     "I 'm a dog".split(),
    ...     "This is my friends ' cat ( the tabby )".split(),
    ... ))], []) # doctest: +NORMALIZE_WHITESPACE +SKIP
    [Tree('ROOT', [Tree('S', [Tree('NP', [Tree('PRP', ['I'])]), Tree('VP', [Tree('VBP', ["'m"]),
    Tree('NP', [Tree('DT', ['a']), Tree('NN', ['dog'])])])])]), Tree('ROOT', [Tree('S', [Tree('NP',
    [Tree('DT', ['This'])]), Tree('VP', [Tree('VBZ', ['is']), Tree('NP', [Tree('NP', [Tree('NP', [Tree('PRP$', ['my']),
    Tree('NNS', ['friends']), Tree('POS', ["'"])]), Tree('NN', ['cat'])]), Tree('PRN', [Tree('-LRB-', [Tree('', []),
    Tree('NP', [Tree('DT', ['the']), Tree('NN', ['tabby'])]), Tree('-RRB-', [])])])])])])])]

    >>> sum([list(dep_graphs) for dep_graphs in parser.tagged_parse_sents((
    ...     (
    ...         ("The", "DT"),
    ...         ("quick", "JJ"),
    ...         ("brown", "JJ"),
    ...         ("fox", "NN"),
    ...         ("jumped", "VBD"),
    ...         ("over", "IN"),
    ...         ("the", "DT"),
    ...         ("lazy", "JJ"),
    ...         ("dog", "NN"),
    ...         (".", "."),
    ...     ),
    ... ))],[]) # doctest: +NORMALIZE_WHITESPACE +SKIP
    [Tree('ROOT', [Tree('S', [Tree('NP', [Tree('DT', ['The']), Tree('JJ', ['quick']), Tree('JJ', ['brown']),
    Tree('NN', ['fox'])]), Tree('VP', [Tree('VBD', ['jumped']), Tree('PP', [Tree('IN', ['over']), Tree('NP',
    [Tree('DT', ['the']), Tree('JJ', ['lazy']), Tree('NN', ['dog'])])])]), Tree('.', ['.'])])])]
    """
    _OUTPUT_FORMAT = 'penn'

    def __init__(self, *args, **kwargs):
        warnings.warn('The StanfordParser will be deprecated\nPlease use \x1b[91mnltk.parse.corenlp.CoreNLPParser\x1b[0m instead.', DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)

    def _make_tree(self, result):
        return Tree.fromstring(result)