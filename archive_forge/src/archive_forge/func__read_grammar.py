import re
import regex
from nltk.chunk.api import ChunkParserI
from nltk.tree import Tree
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