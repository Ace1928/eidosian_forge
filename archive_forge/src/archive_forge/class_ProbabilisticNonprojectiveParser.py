import logging
import math
from nltk.parse.dependencygraph import DependencyGraph
class ProbabilisticNonprojectiveParser:
    """A probabilistic non-projective dependency parser.

    Nonprojective dependencies allows for "crossing branches" in the parse tree
    which is necessary for representing particular linguistic phenomena, or even
    typical parses in some languages.  This parser follows the MST parsing
    algorithm, outlined in McDonald(2005), which likens the search for the best
    non-projective parse to finding the maximum spanning tree in a weighted
    directed graph.

    >>> class Scorer(DependencyScorerI):
    ...     def train(self, graphs):
    ...         pass
    ...
    ...     def score(self, graph):
    ...         return [
    ...             [[], [5],  [1],  [1]],
    ...             [[], [],   [11], [4]],
    ...             [[], [10], [],   [5]],
    ...             [[], [8],  [8],  []],
    ...         ]


    >>> npp = ProbabilisticNonprojectiveParser()
    >>> npp.train([], Scorer())

    >>> parses = npp.parse(['v1', 'v2', 'v3'], [None, None, None])
    >>> len(list(parses))
    1

    Rule based example

    >>> from nltk.grammar import DependencyGrammar

    >>> grammar = DependencyGrammar.fromstring('''
    ... 'taught' -> 'play' | 'man'
    ... 'man' -> 'the' | 'in'
    ... 'in' -> 'corner'
    ... 'corner' -> 'the'
    ... 'play' -> 'golf' | 'dachshund' | 'to'
    ... 'dachshund' -> 'his'
    ... ''')

    >>> ndp = NonprojectiveDependencyParser(grammar)
    >>> parses = ndp.parse(['the', 'man', 'in', 'the', 'corner', 'taught', 'his', 'dachshund', 'to', 'play', 'golf'])
    >>> len(list(parses))
    4

    """

    def __init__(self):
        """
        Creates a new non-projective parser.
        """
        logging.debug('initializing prob. nonprojective...')

    def train(self, graphs, dependency_scorer):
        """
        Trains a ``DependencyScorerI`` from a set of ``DependencyGraph`` objects,
        and establishes this as the parser's scorer.  This is used to
        initialize the scores on a ``DependencyGraph`` during the parsing
        procedure.

        :type graphs: list(DependencyGraph)
        :param graphs: A list of dependency graphs to train the scorer.
        :type dependency_scorer: DependencyScorerI
        :param dependency_scorer: A scorer which implements the
            ``DependencyScorerI`` interface.
        """
        self._scorer = dependency_scorer
        self._scorer.train(graphs)

    def initialize_edge_scores(self, graph):
        """
        Assigns a score to every edge in the ``DependencyGraph`` graph.
        These scores are generated via the parser's scorer which
        was assigned during the training process.

        :type graph: DependencyGraph
        :param graph: A dependency graph to assign scores to.
        """
        self.scores = self._scorer.score(graph)

    def collapse_nodes(self, new_node, cycle_path, g_graph, b_graph, c_graph):
        """
        Takes a list of nodes that have been identified to belong to a cycle,
        and collapses them into on larger node.  The arcs of all nodes in
        the graph must be updated to account for this.

        :type new_node: Node.
        :param new_node: A Node (Dictionary) to collapse the cycle nodes into.
        :type cycle_path: A list of integers.
        :param cycle_path: A list of node addresses, each of which is in the cycle.
        :type g_graph, b_graph, c_graph: DependencyGraph
        :param g_graph, b_graph, c_graph: Graphs which need to be updated.
        """
        logger.debug('Collapsing nodes...')
        for cycle_node_index in cycle_path:
            g_graph.remove_by_address(cycle_node_index)
        g_graph.add_node(new_node)
        g_graph.redirect_arcs(cycle_path, new_node['address'])

    def update_edge_scores(self, new_node, cycle_path):
        """
        Updates the edge scores to reflect a collapse operation into
        new_node.

        :type new_node: A Node.
        :param new_node: The node which cycle nodes are collapsed into.
        :type cycle_path: A list of integers.
        :param cycle_path: A list of node addresses that belong to the cycle.
        """
        logger.debug('cycle %s', cycle_path)
        cycle_path = self.compute_original_indexes(cycle_path)
        logger.debug('old cycle %s', cycle_path)
        logger.debug('Prior to update: %s', self.scores)
        for i, row in enumerate(self.scores):
            for j, column in enumerate(self.scores[i]):
                logger.debug(self.scores[i][j])
                if j in cycle_path and i not in cycle_path and self.scores[i][j]:
                    subtract_val = self.compute_max_subtract_score(j, cycle_path)
                    logger.debug('%s - %s', self.scores[i][j], subtract_val)
                    new_vals = []
                    for cur_val in self.scores[i][j]:
                        new_vals.append(cur_val - subtract_val)
                    self.scores[i][j] = new_vals
        for i, row in enumerate(self.scores):
            for j, cell in enumerate(self.scores[i]):
                if i in cycle_path and j in cycle_path:
                    self.scores[i][j] = []
        logger.debug('After update: %s', self.scores)

    def compute_original_indexes(self, new_indexes):
        """
        As nodes are collapsed into others, they are replaced
        by the new node in the graph, but it's still necessary
        to keep track of what these original nodes were.  This
        takes a list of node addresses and replaces any collapsed
        node addresses with their original addresses.

        :type new_indexes: A list of integers.
        :param new_indexes: A list of node addresses to check for
            subsumed nodes.
        """
        swapped = True
        while swapped:
            originals = []
            swapped = False
            for new_index in new_indexes:
                if new_index in self.inner_nodes:
                    for old_val in self.inner_nodes[new_index]:
                        if old_val not in originals:
                            originals.append(old_val)
                            swapped = True
                else:
                    originals.append(new_index)
            new_indexes = originals
        return new_indexes

    def compute_max_subtract_score(self, column_index, cycle_indexes):
        """
        When updating scores the score of the highest-weighted incoming
        arc is subtracted upon collapse.  This returns the correct
        amount to subtract from that edge.

        :type column_index: integer.
        :param column_index: A index representing the column of incoming arcs
            to a particular node being updated
        :type cycle_indexes: A list of integers.
        :param cycle_indexes: Only arcs from cycle nodes are considered.  This
            is a list of such nodes addresses.
        """
        max_score = -100000
        for row_index in cycle_indexes:
            for subtract_val in self.scores[row_index][column_index]:
                if subtract_val > max_score:
                    max_score = subtract_val
        return max_score

    def best_incoming_arc(self, node_index):
        """
        Returns the source of the best incoming arc to the
        node with address: node_index

        :type node_index: integer.
        :param node_index: The address of the 'destination' node,
            the node that is arced to.
        """
        originals = self.compute_original_indexes([node_index])
        logger.debug('originals: %s', originals)
        max_arc = None
        max_score = None
        for row_index in range(len(self.scores)):
            for col_index in range(len(self.scores[row_index])):
                if col_index in originals and (max_score is None or self.scores[row_index][col_index] > max_score):
                    max_score = self.scores[row_index][col_index]
                    max_arc = row_index
                    logger.debug('%s, %s', row_index, col_index)
        logger.debug(max_score)
        for key in self.inner_nodes:
            replaced_nodes = self.inner_nodes[key]
            if max_arc in replaced_nodes:
                return key
        return max_arc

    def original_best_arc(self, node_index):
        originals = self.compute_original_indexes([node_index])
        max_arc = None
        max_score = None
        max_orig = None
        for row_index in range(len(self.scores)):
            for col_index in range(len(self.scores[row_index])):
                if col_index in originals and (max_score is None or self.scores[row_index][col_index] > max_score):
                    max_score = self.scores[row_index][col_index]
                    max_arc = row_index
                    max_orig = col_index
        return [max_arc, max_orig]

    def parse(self, tokens, tags):
        """
        Parses a list of tokens in accordance to the MST parsing algorithm
        for non-projective dependency parses.  Assumes that the tokens to
        be parsed have already been tagged and those tags are provided.  Various
        scoring methods can be used by implementing the ``DependencyScorerI``
        interface and passing it to the training algorithm.

        :type tokens: list(str)
        :param tokens: A list of words or punctuation to be parsed.
        :type tags: list(str)
        :param tags: A list of tags corresponding by index to the words in the tokens list.
        :return: An iterator of non-projective parses.
        :rtype: iter(DependencyGraph)
        """
        self.inner_nodes = {}
        g_graph = DependencyGraph()
        for index, token in enumerate(tokens):
            g_graph.nodes[index + 1].update({'word': token, 'tag': tags[index], 'rel': 'NTOP', 'address': index + 1})
        g_graph.connect_graph()
        original_graph = DependencyGraph()
        for index, token in enumerate(tokens):
            original_graph.nodes[index + 1].update({'word': token, 'tag': tags[index], 'rel': 'NTOP', 'address': index + 1})
        b_graph = DependencyGraph()
        c_graph = DependencyGraph()
        for index, token in enumerate(tokens):
            c_graph.nodes[index + 1].update({'word': token, 'tag': tags[index], 'rel': 'NTOP', 'address': index + 1})
        self.initialize_edge_scores(g_graph)
        logger.debug(self.scores)
        unvisited_vertices = [vertex['address'] for vertex in c_graph.nodes.values()]
        nr_vertices = len(tokens)
        betas = {}
        while unvisited_vertices:
            current_vertex = unvisited_vertices.pop(0)
            logger.debug('current_vertex: %s', current_vertex)
            current_node = g_graph.get_by_address(current_vertex)
            logger.debug('current_node: %s', current_node)
            best_in_edge = self.best_incoming_arc(current_vertex)
            betas[current_vertex] = self.original_best_arc(current_vertex)
            logger.debug('best in arc: %s --> %s', best_in_edge, current_vertex)
            for new_vertex in [current_vertex, best_in_edge]:
                b_graph.nodes[new_vertex].update({'word': 'TEMP', 'rel': 'NTOP', 'address': new_vertex})
            b_graph.add_arc(best_in_edge, current_vertex)
            cycle_path = b_graph.contains_cycle()
            if cycle_path:
                new_node = {'word': 'NONE', 'rel': 'NTOP', 'address': nr_vertices + 1}
                c_graph.add_node(new_node)
                self.update_edge_scores(new_node, cycle_path)
                self.collapse_nodes(new_node, cycle_path, g_graph, b_graph, c_graph)
                for cycle_index in cycle_path:
                    c_graph.add_arc(new_node['address'], cycle_index)
                self.inner_nodes[new_node['address']] = cycle_path
                unvisited_vertices.insert(0, nr_vertices + 1)
                nr_vertices += 1
                for cycle_node_address in cycle_path:
                    b_graph.remove_by_address(cycle_node_address)
            logger.debug('g_graph: %s', g_graph)
            logger.debug('b_graph: %s', b_graph)
            logger.debug('c_graph: %s', c_graph)
            logger.debug('Betas: %s', betas)
            logger.debug('replaced nodes %s', self.inner_nodes)
        logger.debug('Final scores: %s', self.scores)
        logger.debug('Recovering parse...')
        for i in range(len(tokens) + 1, nr_vertices + 1):
            betas[betas[i][1]] = betas[i]
        logger.debug('Betas: %s', betas)
        for node in original_graph.nodes.values():
            node['deps'] = {}
        for i in range(1, len(tokens) + 1):
            original_graph.add_arc(betas[i][0], betas[i][1])
        logger.debug('Done.')
        yield original_graph