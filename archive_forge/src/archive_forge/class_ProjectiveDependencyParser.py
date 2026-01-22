from collections import defaultdict
from functools import total_ordering
from itertools import chain
from nltk.grammar import (
from nltk.internals import raise_unorderable_types
from nltk.parse.dependencygraph import DependencyGraph
class ProjectiveDependencyParser:
    """
    A projective, rule-based, dependency parser.  A ProjectiveDependencyParser
    is created with a DependencyGrammar, a set of productions specifying
    word-to-word dependency relations.  The parse() method will then
    return the set of all parses, in tree representation, for a given input
    sequence of tokens.  Each parse must meet the requirements of the both
    the grammar and the projectivity constraint which specifies that the
    branches of the dependency tree are not allowed to cross.  Alternatively,
    this can be understood as stating that each parent node and its children
    in the parse tree form a continuous substring of the input sequence.
    """

    def __init__(self, dependency_grammar):
        """
        Create a new ProjectiveDependencyParser, from a word-to-word
        dependency grammar ``DependencyGrammar``.

        :param dependency_grammar: A word-to-word relation dependencygrammar.
        :type dependency_grammar: DependencyGrammar
        """
        self._grammar = dependency_grammar

    def parse(self, tokens):
        """
        Performs a projective dependency parse on the list of tokens using
        a chart-based, span-concatenation algorithm similar to Eisner (1996).

        :param tokens: The list of input tokens.
        :type tokens: list(str)
        :return: An iterator over parse trees.
        :rtype: iter(Tree)
        """
        self._tokens = list(tokens)
        chart = []
        for i in range(0, len(self._tokens) + 1):
            chart.append([])
            for j in range(0, len(self._tokens) + 1):
                chart[i].append(ChartCell(i, j))
                if i == j + 1:
                    chart[i][j].add(DependencySpan(i - 1, i, i - 1, [-1], ['null']))
        for i in range(1, len(self._tokens) + 1):
            for j in range(i - 2, -1, -1):
                for k in range(i - 1, j, -1):
                    for span1 in chart[k][j]._entries:
                        for span2 in chart[i][k]._entries:
                            for newspan in self.concatenate(span1, span2):
                                chart[i][j].add(newspan)
        for parse in chart[len(self._tokens)][0]._entries:
            conll_format = ''
            for i in range(len(tokens)):
                conll_format += '\t%d\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s\n' % (i + 1, tokens[i], tokens[i], 'null', 'null', 'null', parse._arcs[i] + 1, 'ROOT', '-', '-')
            dg = DependencyGraph(conll_format)
            yield dg.tree()

    def concatenate(self, span1, span2):
        """
        Concatenates the two spans in whichever way possible.  This
        includes rightward concatenation (from the leftmost word of the
        leftmost span to the rightmost word of the rightmost span) and
        leftward concatenation (vice-versa) between adjacent spans.  Unlike
        Eisner's presentation of span concatenation, these spans do not
        share or pivot on a particular word/word-index.

        :return: A list of new spans formed through concatenation.
        :rtype: list(DependencySpan)
        """
        spans = []
        if span1._start_index == span2._start_index:
            print('Error: Mismatched spans - replace this with thrown error')
        if span1._start_index > span2._start_index:
            temp_span = span1
            span1 = span2
            span2 = temp_span
        new_arcs = span1._arcs + span2._arcs
        new_tags = span1._tags + span2._tags
        if self._grammar.contains(self._tokens[span1._head_index], self._tokens[span2._head_index]):
            new_arcs[span2._head_index - span1._start_index] = span1._head_index
            spans.append(DependencySpan(span1._start_index, span2._end_index, span1._head_index, new_arcs, new_tags))
        new_arcs = span1._arcs + span2._arcs
        if self._grammar.contains(self._tokens[span2._head_index], self._tokens[span1._head_index]):
            new_arcs[span1._head_index - span1._start_index] = span2._head_index
            spans.append(DependencySpan(span1._start_index, span2._end_index, span2._head_index, new_arcs, new_tags))
        return spans