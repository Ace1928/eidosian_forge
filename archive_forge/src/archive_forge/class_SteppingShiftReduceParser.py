from nltk.grammar import Nonterminal
from nltk.parse.api import ParserI
from nltk.tree import Tree
class SteppingShiftReduceParser(ShiftReduceParser):
    """
    A ``ShiftReduceParser`` that allows you to setp through the parsing
    process, performing a single operation at a time.  It also allows
    you to change the parser's grammar midway through parsing a text.

    The ``initialize`` method is used to start parsing a text.
    ``shift`` performs a single shift operation, and ``reduce`` performs
    a single reduce operation.  ``step`` will perform a single reduce
    operation if possible; otherwise, it will perform a single shift
    operation.  ``parses`` returns the set of parses that have been
    found by the parser.

    :ivar _history: A list of ``(stack, remaining_text)`` pairs,
        containing all of the previous states of the parser.  This
        history is used to implement the ``undo`` operation.
    :see: ``nltk.grammar``
    """

    def __init__(self, grammar, trace=0):
        super().__init__(grammar, trace)
        self._stack = None
        self._remaining_text = None
        self._history = []

    def parse(self, tokens):
        tokens = list(tokens)
        self.initialize(tokens)
        while self.step():
            pass
        return self.parses()

    def stack(self):
        """
        :return: The parser's stack.
        :rtype: list(str and Tree)
        """
        return self._stack

    def remaining_text(self):
        """
        :return: The portion of the text that is not yet covered by the
            stack.
        :rtype: list(str)
        """
        return self._remaining_text

    def initialize(self, tokens):
        """
        Start parsing a given text.  This sets the parser's stack to
        ``[]`` and sets its remaining text to ``tokens``.
        """
        self._stack = []
        self._remaining_text = tokens
        self._history = []

    def step(self):
        """
        Perform a single parsing operation.  If a reduction is
        possible, then perform that reduction, and return the
        production that it is based on.  Otherwise, if a shift is
        possible, then perform it, and return True.  Otherwise,
        return False.

        :return: False if no operation was performed; True if a shift was
            performed; and the CFG production used to reduce if a
            reduction was performed.
        :rtype: Production or bool
        """
        return self.reduce() or self.shift()

    def shift(self):
        """
        Move a token from the beginning of the remaining text to the
        end of the stack.  If there are no more tokens in the
        remaining text, then do nothing.

        :return: True if the shift operation was successful.
        :rtype: bool
        """
        if len(self._remaining_text) == 0:
            return False
        self._history.append((self._stack[:], self._remaining_text[:]))
        self._shift(self._stack, self._remaining_text)
        return True

    def reduce(self, production=None):
        """
        Use ``production`` to combine the rightmost stack elements into
        a single Tree.  If ``production`` does not match the
        rightmost stack elements, then do nothing.

        :return: The production used to reduce the stack, if a
            reduction was performed.  If no reduction was performed,
            return None.

        :rtype: Production or None
        """
        self._history.append((self._stack[:], self._remaining_text[:]))
        return_val = self._reduce(self._stack, self._remaining_text, production)
        if not return_val:
            self._history.pop()
        return return_val

    def undo(self):
        """
        Return the parser to its state before the most recent
        shift or reduce operation.  Calling ``undo`` repeatedly return
        the parser to successively earlier states.  If no shift or
        reduce operations have been performed, ``undo`` will make no
        changes.

        :return: true if an operation was successfully undone.
        :rtype: bool
        """
        if len(self._history) == 0:
            return False
        self._stack, self._remaining_text = self._history.pop()
        return True

    def reducible_productions(self):
        """
        :return: A list of the productions for which reductions are
            available for the current parser state.
        :rtype: list(Production)
        """
        productions = []
        for production in self._grammar.productions():
            rhslen = len(production.rhs())
            if self._match_rhs(production.rhs(), self._stack[-rhslen:]):
                productions.append(production)
        return productions

    def parses(self):
        """
        :return: An iterator of the parses that have been found by this
            parser so far.
        :rtype: iter(Tree)
        """
        if len(self._remaining_text) == 0 and len(self._stack) == 1 and (self._stack[0].label() == self._grammar.start().symbol()):
            yield self._stack[0]

    def set_grammar(self, grammar):
        """
        Change the grammar used to parse texts.

        :param grammar: The new grammar.
        :type grammar: CFG
        """
        self._grammar = grammar