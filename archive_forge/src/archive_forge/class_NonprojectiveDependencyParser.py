import logging
import math
from nltk.parse.dependencygraph import DependencyGraph
class NonprojectiveDependencyParser:
    """
    A non-projective, rule-based, dependency parser.  This parser
    will return the set of all possible non-projective parses based on
    the word-to-word relations defined in the parser's dependency
    grammar, and will allow the branches of the parse tree to cross
    in order to capture a variety of linguistic phenomena that a
    projective parser will not.
    """

    def __init__(self, dependency_grammar):
        """
        Creates a new ``NonprojectiveDependencyParser``.

        :param dependency_grammar: a grammar of word-to-word relations.
        :type dependency_grammar: DependencyGrammar
        """
        self._grammar = dependency_grammar

    def parse(self, tokens):
        """
        Parses the input tokens with respect to the parser's grammar.  Parsing
        is accomplished by representing the search-space of possible parses as
        a fully-connected directed graph.  Arcs that would lead to ungrammatical
        parses are removed and a lattice is constructed of length n, where n is
        the number of input tokens, to represent all possible grammatical
        traversals.  All possible paths through the lattice are then enumerated
        to produce the set of non-projective parses.

        param tokens: A list of tokens to parse.
        type tokens: list(str)
        return: An iterator of non-projective parses.
        rtype: iter(DependencyGraph)
        """
        self._graph = DependencyGraph()
        for index, token in enumerate(tokens):
            self._graph.nodes[index] = {'word': token, 'deps': [], 'rel': 'NTOP', 'address': index}
        for head_node in self._graph.nodes.values():
            deps = []
            for dep_node in self._graph.nodes.values():
                if self._grammar.contains(head_node['word'], dep_node['word']) and head_node['word'] != dep_node['word']:
                    deps.append(dep_node['address'])
            head_node['deps'] = deps
        roots = []
        possible_heads = []
        for i, word in enumerate(tokens):
            heads = []
            for j, head in enumerate(tokens):
                if i != j and self._grammar.contains(head, word):
                    heads.append(j)
            if len(heads) == 0:
                roots.append(i)
            possible_heads.append(heads)
        if len(roots) < 2:
            if len(roots) == 0:
                for i in range(len(tokens)):
                    roots.append(i)
            analyses = []
            for _ in roots:
                stack = []
                analysis = [[] for i in range(len(possible_heads))]
            i = 0
            forward = True
            while i >= 0:
                if forward:
                    if len(possible_heads[i]) == 1:
                        analysis[i] = possible_heads[i][0]
                    elif len(possible_heads[i]) == 0:
                        analysis[i] = -1
                    else:
                        head = possible_heads[i].pop()
                        analysis[i] = head
                        stack.append([i, head])
                if not forward:
                    index_on_stack = False
                    for stack_item in stack:
                        if stack_item[0] == i:
                            index_on_stack = True
                    orig_length = len(possible_heads[i])
                    if index_on_stack and orig_length == 0:
                        for j in range(len(stack) - 1, -1, -1):
                            stack_item = stack[j]
                            if stack_item[0] == i:
                                possible_heads[i].append(stack.pop(j)[1])
                    elif index_on_stack and orig_length > 0:
                        head = possible_heads[i].pop()
                        analysis[i] = head
                        stack.append([i, head])
                        forward = True
                if i + 1 == len(possible_heads):
                    analyses.append(analysis[:])
                    forward = False
                if forward:
                    i += 1
                else:
                    i -= 1
        for analysis in analyses:
            if analysis.count(-1) > 1:
                continue
            graph = DependencyGraph()
            graph.root = graph.nodes[analysis.index(-1) + 1]
            for address, (token, head_index) in enumerate(zip(tokens, analysis), start=1):
                head_address = head_index + 1
                node = graph.nodes[address]
                node.update({'word': token, 'address': address})
                if head_address == 0:
                    rel = 'ROOT'
                else:
                    rel = ''
                graph.nodes[head_index + 1]['deps'][rel].append(address)
            yield graph