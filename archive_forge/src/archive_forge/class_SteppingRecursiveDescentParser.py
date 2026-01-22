from nltk.grammar import Nonterminal
from nltk.parse.api import ParserI
from nltk.tree import ImmutableTree, Tree
class SteppingRecursiveDescentParser(RecursiveDescentParser):
    """
    A ``RecursiveDescentParser`` that allows you to step through the
    parsing process, performing a single operation at a time.

    The ``initialize`` method is used to start parsing a text.
    ``expand`` expands the first element on the frontier using a single
    CFG production, and ``match`` matches the first element on the
    frontier against the next text token. ``backtrack`` undoes the most
    recent expand or match operation.  ``step`` performs a single
    expand, match, or backtrack operation.  ``parses`` returns the set
    of parses that have been found by the parser.

    :ivar _history: A list of ``(rtext, tree, frontier)`` tripples,
        containing the previous states of the parser.  This history is
        used to implement the ``backtrack`` operation.
    :ivar _tried_e: A record of all productions that have been tried
        for a given tree.  This record is used by ``expand`` to perform
        the next untried production.
    :ivar _tried_m: A record of what tokens have been matched for a
        given tree.  This record is used by ``step`` to decide whether
        or not to match a token.
    :see: ``nltk.grammar``
    """

    def __init__(self, grammar, trace=0):
        super().__init__(grammar, trace)
        self._rtext = None
        self._tree = None
        self._frontier = [()]
        self._tried_e = {}
        self._tried_m = {}
        self._history = []
        self._parses = []

    def _freeze(self, tree):
        c = tree.copy()
        return ImmutableTree.convert(c)

    def parse(self, tokens):
        tokens = list(tokens)
        self.initialize(tokens)
        while self.step() is not None:
            pass
        return self.parses()

    def initialize(self, tokens):
        """
        Start parsing a given text.  This sets the parser's tree to
        the start symbol, its frontier to the root node, and its
        remaining text to ``token['SUBTOKENS']``.
        """
        self._rtext = tokens
        start = self._grammar.start().symbol()
        self._tree = Tree(start, [])
        self._frontier = [()]
        self._tried_e = {}
        self._tried_m = {}
        self._history = []
        self._parses = []
        if self._trace:
            self._trace_start(self._tree, self._frontier, self._rtext)

    def remaining_text(self):
        """
        :return: The portion of the text that is not yet covered by the
            tree.
        :rtype: list(str)
        """
        return self._rtext

    def frontier(self):
        """
        :return: A list of the tree locations of all subtrees that
            have not yet been expanded, and all leaves that have not
            yet been matched.
        :rtype: list(tuple(int))
        """
        return self._frontier

    def tree(self):
        """
        :return: A partial structure for the text that is
            currently being parsed.  The elements specified by the
            frontier have not yet been expanded or matched.
        :rtype: Tree
        """
        return self._tree

    def step(self):
        """
        Perform a single parsing operation.  If an untried match is
        possible, then perform the match, and return the matched
        token.  If an untried expansion is possible, then perform the
        expansion, and return the production that it is based on.  If
        backtracking is possible, then backtrack, and return True.
        Otherwise, return None.

        :return: None if no operation was performed; a token if a match
            was performed; a production if an expansion was performed;
            and True if a backtrack operation was performed.
        :rtype: Production or String or bool
        """
        if self.untried_match():
            token = self.match()
            if token is not None:
                return token
        production = self.expand()
        if production is not None:
            return production
        if self.backtrack():
            self._trace_backtrack(self._tree, self._frontier)
            return True
        return None

    def expand(self, production=None):
        """
        Expand the first element of the frontier.  In particular, if
        the first element of the frontier is a subtree whose node type
        is equal to ``production``'s left hand side, then add a child
        to that subtree for each element of ``production``'s right hand
        side.  If ``production`` is not specified, then use the first
        untried expandable production.  If all expandable productions
        have been tried, do nothing.

        :return: The production used to expand the frontier, if an
           expansion was performed.  If no expansion was performed,
           return None.
        :rtype: Production or None
        """
        if len(self._frontier) == 0:
            return None
        if not isinstance(self._tree[self._frontier[0]], Tree):
            return None
        if production is None:
            productions = self.untried_expandable_productions()
        else:
            productions = [production]
        parses = []
        for prod in productions:
            self._tried_e.setdefault(self._freeze(self._tree), []).append(prod)
            for _result in self._expand(self._rtext, self._tree, self._frontier, prod):
                return prod
        return None

    def match(self):
        """
        Match the first element of the frontier.  In particular, if
        the first element of the frontier has the same type as the
        next text token, then substitute the text token into the tree.

        :return: The token matched, if a match operation was
            performed.  If no match was performed, return None
        :rtype: str or None
        """
        tok = self._rtext[0]
        self._tried_m.setdefault(self._freeze(self._tree), []).append(tok)
        if len(self._frontier) == 0:
            return None
        if isinstance(self._tree[self._frontier[0]], Tree):
            return None
        for _result in self._match(self._rtext, self._tree, self._frontier):
            return self._history[-1][0][0]
        return None

    def backtrack(self):
        """
        Return the parser to its state before the most recent
        match or expand operation.  Calling ``undo`` repeatedly return
        the parser to successively earlier states.  If no match or
        expand operations have been performed, ``undo`` will make no
        changes.

        :return: true if an operation was successfully undone.
        :rtype: bool
        """
        if len(self._history) == 0:
            return False
        self._rtext, self._tree, self._frontier = self._history.pop()
        return True

    def expandable_productions(self):
        """
        :return: A list of all the productions for which expansions
            are available for the current parser state.
        :rtype: list(Production)
        """
        if len(self._frontier) == 0:
            return []
        frontier_child = self._tree[self._frontier[0]]
        if len(self._frontier) == 0 or not isinstance(frontier_child, Tree):
            return []
        return [p for p in self._grammar.productions() if p.lhs().symbol() == frontier_child.label()]

    def untried_expandable_productions(self):
        """
        :return: A list of all the untried productions for which
            expansions are available for the current parser state.
        :rtype: list(Production)
        """
        tried_expansions = self._tried_e.get(self._freeze(self._tree), [])
        return [p for p in self.expandable_productions() if p not in tried_expansions]

    def untried_match(self):
        """
        :return: Whether the first element of the frontier is a token
            that has not yet been matched.
        :rtype: bool
        """
        if len(self._rtext) == 0:
            return False
        tried_matches = self._tried_m.get(self._freeze(self._tree), [])
        return self._rtext[0] not in tried_matches

    def currently_complete(self):
        """
        :return: Whether the parser's current state represents a
            complete parse.
        :rtype: bool
        """
        return len(self._frontier) == 0 and len(self._rtext) == 0

    def _parse(self, remaining_text, tree, frontier):
        """
        A stub version of ``_parse`` that sets the parsers current
        state to the given arguments.  In ``RecursiveDescentParser``,
        the ``_parse`` method is used to recursively continue parsing a
        text.  ``SteppingRecursiveDescentParser`` overrides it to
        capture these recursive calls.  It records the parser's old
        state in the history (to allow for backtracking), and updates
        the parser's new state using the given arguments.  Finally, it
        returns ``[1]``, which is used by ``match`` and ``expand`` to
        detect whether their operations were successful.

        :return: ``[1]``
        :rtype: list of int
        """
        self._history.append((self._rtext, self._tree, self._frontier))
        self._rtext = remaining_text
        self._tree = tree
        self._frontier = frontier
        if len(frontier) == 0 and len(remaining_text) == 0:
            self._parses.append(tree)
            self._trace_succeed(self._tree, self._frontier)
        return [1]

    def parses(self):
        """
        :return: An iterator of the parses that have been found by this
            parser so far.
        :rtype: list of Tree
        """
        return iter(self._parses)

    def set_grammar(self, grammar):
        """
        Change the grammar used to parse texts.

        :param grammar: The new grammar.
        :type grammar: CFG
        """
        self._grammar = grammar