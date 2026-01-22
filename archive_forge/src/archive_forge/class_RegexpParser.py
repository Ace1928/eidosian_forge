import re
import regex
from nltk.chunk.api import ChunkParserI
from nltk.tree import Tree
class RegexpParser(ChunkParserI):
    """
    A grammar based chunk parser.  ``chunk.RegexpParser`` uses a set of
    regular expression patterns to specify the behavior of the parser.
    The chunking of the text is encoded using a ``ChunkString``, and
    each rule acts by modifying the chunking in the ``ChunkString``.
    The rules are all implemented using regular expression matching
    and substitution.

    A grammar contains one or more clauses in the following form::

     NP:
       {<DT|JJ>}          # chunk determiners and adjectives
       }<[\\.VI].*>+{      # strip any tag beginning with V, I, or .
       <.*>}{<DT>         # split a chunk at a determiner
       <DT|JJ>{}<NN.*>    # merge chunk ending with det/adj
                          # with one starting with a noun

    The patterns of a clause are executed in order.  An earlier
    pattern may introduce a chunk boundary that prevents a later
    pattern from executing.  Sometimes an individual pattern will
    match on multiple, overlapping extents of the input.  As with
    regular expression substitution more generally, the chunker will
    identify the first match possible, then continue looking for matches
    after this one has ended.

    The clauses of a grammar are also executed in order.  A cascaded
    chunk parser is one having more than one clause.  The maximum depth
    of a parse tree created by this chunk parser is the same as the
    number of clauses in the grammar.

    When tracing is turned on, the comment portion of a line is displayed
    each time the corresponding pattern is applied.

    :type _start: str
    :ivar _start: The start symbol of the grammar (the root node of
        resulting trees)
    :type _stages: int
    :ivar _stages: The list of parsing stages corresponding to the grammar

    """

    def __init__(self, grammar, root_label='S', loop=1, trace=0):
        """
        Create a new chunk parser, from the given start state
        and set of chunk patterns.

        :param grammar: The grammar, or a list of RegexpChunkParser objects
        :type grammar: str or list(RegexpChunkParser)
        :param root_label: The top node of the tree being created
        :type root_label: str or Nonterminal
        :param loop: The number of times to run through the patterns
        :type loop: int
        :type trace: int
        :param trace: The level of tracing that should be used when
            parsing a text.  ``0`` will generate no tracing output;
            ``1`` will generate normal tracing output; and ``2`` or
            higher will generate verbose tracing output.
        """
        self._trace = trace
        self._stages = []
        self._grammar = grammar
        self._loop = loop
        if isinstance(grammar, str):
            self._read_grammar(grammar, root_label, trace)
        else:
            type_err = 'Expected string or list of RegexpChunkParsers for the grammar.'
            try:
                grammar = list(grammar)
            except BaseException as e:
                raise TypeError(type_err) from e
            for elt in grammar:
                if not isinstance(elt, RegexpChunkParser):
                    raise TypeError(type_err)
            self._stages = grammar

    def _read_grammar(self, grammar, root_label, trace):
        """
        Helper function for __init__: read the grammar if it is a
        string.
        """
        rules = []
        lhs = None
        pattern = regex.compile('(?P<nonterminal>(\\.|[^:])*)(:(?P<rule>.*))')
        for line in grammar.split('\n'):
            line = line.strip()
            m = pattern.match(line)
            if m:
                self._add_stage(rules, lhs, root_label, trace)
                lhs = m.group('nonterminal').strip()
                rules = []
                line = m.group('rule').strip()
            if line == '' or line.startswith('#'):
                continue
            rules.append(RegexpChunkRule.fromstring(line))
        self._add_stage(rules, lhs, root_label, trace)

    def _add_stage(self, rules, lhs, root_label, trace):
        """
        Helper function for __init__: add a new stage to the parser.
        """
        if rules != []:
            if not lhs:
                raise ValueError('Expected stage marker (eg NP:)')
            parser = RegexpChunkParser(rules, chunk_label=lhs, root_label=root_label, trace=trace)
            self._stages.append(parser)

    def parse(self, chunk_struct, trace=None):
        """
        Apply the chunk parser to this input.

        :type chunk_struct: Tree
        :param chunk_struct: the chunk structure to be (further) chunked
            (this tree is modified, and is also returned)
        :type trace: int
        :param trace: The level of tracing that should be used when
            parsing a text.  ``0`` will generate no tracing output;
            ``1`` will generate normal tracing output; and ``2`` or
            higher will generate verbose tracing output.  This value
            overrides the trace level value that was given to the
            constructor.
        :return: the chunked output.
        :rtype: Tree
        """
        if trace is None:
            trace = self._trace
        for i in range(self._loop):
            for parser in self._stages:
                chunk_struct = parser.parse(chunk_struct, trace=trace)
        return chunk_struct

    def __repr__(self):
        """
        :return: a concise string representation of this ``chunk.RegexpParser``.
        :rtype: str
        """
        return '<chunk.RegexpParser with %d stages>' % len(self._stages)

    def __str__(self):
        """
        :return: a verbose string representation of this
            ``RegexpParser``.
        :rtype: str
        """
        s = 'chunk.RegexpParser with %d stages:\n' % len(self._stages)
        margin = 0
        for parser in self._stages:
            s += '%s\n' % parser
        return s[:-1]