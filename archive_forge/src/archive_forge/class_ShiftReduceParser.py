from nltk.grammar import Nonterminal
from nltk.parse.api import ParserI
from nltk.tree import Tree
class ShiftReduceParser(ParserI):
    """
    A simple bottom-up CFG parser that uses two operations, "shift"
    and "reduce", to find a single parse for a text.

    ``ShiftReduceParser`` maintains a stack, which records the
    structure of a portion of the text.  This stack is a list of
    strings and Trees that collectively cover a portion of
    the text.  For example, while parsing the sentence "the dog saw
    the man" with a typical grammar, ``ShiftReduceParser`` will produce
    the following stack, which covers "the dog saw"::

       [(NP: (Det: 'the') (N: 'dog')), (V: 'saw')]

    ``ShiftReduceParser`` attempts to extend the stack to cover the
    entire text, and to combine the stack elements into a single tree,
    producing a complete parse for the sentence.

    Initially, the stack is empty.  It is extended to cover the text,
    from left to right, by repeatedly applying two operations:

      - "shift" moves a token from the beginning of the text to the
        end of the stack.
      - "reduce" uses a CFG production to combine the rightmost stack
        elements into a single Tree.

    Often, more than one operation can be performed on a given stack.
    In this case, ``ShiftReduceParser`` uses the following heuristics
    to decide which operation to perform:

      - Only shift if no reductions are available.
      - If multiple reductions are available, then apply the reduction
        whose CFG production is listed earliest in the grammar.

    Note that these heuristics are not guaranteed to choose an
    operation that leads to a parse of the text.  Also, if multiple
    parses exists, ``ShiftReduceParser`` will return at most one of
    them.

    :see: ``nltk.grammar``
    """

    def __init__(self, grammar, trace=0):
        """
        Create a new ``ShiftReduceParser``, that uses ``grammar`` to
        parse texts.

        :type grammar: Grammar
        :param grammar: The grammar used to parse texts.
        :type trace: int
        :param trace: The level of tracing that should be used when
            parsing a text.  ``0`` will generate no tracing output;
            and higher numbers will produce more verbose tracing
            output.
        """
        self._grammar = grammar
        self._trace = trace
        self._check_grammar()

    def grammar(self):
        return self._grammar

    def parse(self, tokens):
        tokens = list(tokens)
        self._grammar.check_coverage(tokens)
        stack = []
        remaining_text = tokens
        if self._trace:
            print('Parsing %r' % ' '.join(tokens))
            self._trace_stack(stack, remaining_text)
        while len(remaining_text) > 0:
            self._shift(stack, remaining_text)
            while self._reduce(stack, remaining_text):
                pass
        if len(stack) == 1:
            if stack[0].label() == self._grammar.start().symbol():
                yield stack[0]

    def _shift(self, stack, remaining_text):
        """
        Move a token from the beginning of ``remaining_text`` to the
        end of ``stack``.

        :type stack: list(str and Tree)
        :param stack: A list of strings and Trees, encoding
            the structure of the text that has been parsed so far.
        :type remaining_text: list(str)
        :param remaining_text: The portion of the text that is not yet
            covered by ``stack``.
        :rtype: None
        """
        stack.append(remaining_text[0])
        remaining_text.remove(remaining_text[0])
        if self._trace:
            self._trace_shift(stack, remaining_text)

    def _match_rhs(self, rhs, rightmost_stack):
        """
        :rtype: bool
        :return: true if the right hand side of a CFG production
            matches the rightmost elements of the stack.  ``rhs``
            matches ``rightmost_stack`` if they are the same length,
            and each element of ``rhs`` matches the corresponding
            element of ``rightmost_stack``.  A nonterminal element of
            ``rhs`` matches any Tree whose node value is equal
            to the nonterminal's symbol.  A terminal element of ``rhs``
            matches any string whose type is equal to the terminal.
        :type rhs: list(terminal and Nonterminal)
        :param rhs: The right hand side of a CFG production.
        :type rightmost_stack: list(string and Tree)
        :param rightmost_stack: The rightmost elements of the parser's
            stack.
        """
        if len(rightmost_stack) != len(rhs):
            return False
        for i in range(len(rightmost_stack)):
            if isinstance(rightmost_stack[i], Tree):
                if not isinstance(rhs[i], Nonterminal):
                    return False
                if rightmost_stack[i].label() != rhs[i].symbol():
                    return False
            else:
                if isinstance(rhs[i], Nonterminal):
                    return False
                if rightmost_stack[i] != rhs[i]:
                    return False
        return True

    def _reduce(self, stack, remaining_text, production=None):
        """
        Find a CFG production whose right hand side matches the
        rightmost stack elements; and combine those stack elements
        into a single Tree, with the node specified by the
        production's left-hand side.  If more than one CFG production
        matches the stack, then use the production that is listed
        earliest in the grammar.  The new Tree replaces the
        elements in the stack.

        :rtype: Production or None
        :return: If a reduction is performed, then return the CFG
            production that the reduction is based on; otherwise,
            return false.
        :type stack: list(string and Tree)
        :param stack: A list of strings and Trees, encoding
            the structure of the text that has been parsed so far.
        :type remaining_text: list(str)
        :param remaining_text: The portion of the text that is not yet
            covered by ``stack``.
        """
        if production is None:
            productions = self._grammar.productions()
        else:
            productions = [production]
        for production in productions:
            rhslen = len(production.rhs())
            if self._match_rhs(production.rhs(), stack[-rhslen:]):
                tree = Tree(production.lhs().symbol(), stack[-rhslen:])
                stack[-rhslen:] = [tree]
                if self._trace:
                    self._trace_reduce(stack, production, remaining_text)
                return production
        return None

    def trace(self, trace=2):
        """
        Set the level of tracing output that should be generated when
        parsing a text.

        :type trace: int
        :param trace: The trace level.  A trace level of ``0`` will
            generate no tracing output; and higher trace levels will
            produce more verbose tracing output.
        :rtype: None
        """
        self._trace = trace

    def _trace_stack(self, stack, remaining_text, marker=' '):
        """
        Print trace output displaying the given stack and text.

        :rtype: None
        :param marker: A character that is printed to the left of the
            stack.  This is used with trace level 2 to print 'S'
            before shifted stacks and 'R' before reduced stacks.
        """
        s = '  ' + marker + ' [ '
        for elt in stack:
            if isinstance(elt, Tree):
                s += repr(Nonterminal(elt.label())) + ' '
            else:
                s += repr(elt) + ' '
        s += '* ' + ' '.join(remaining_text) + ']'
        print(s)

    def _trace_shift(self, stack, remaining_text):
        """
        Print trace output displaying that a token has been shifted.

        :rtype: None
        """
        if self._trace > 2:
            print('Shift %r:' % stack[-1])
        if self._trace == 2:
            self._trace_stack(stack, remaining_text, 'S')
        elif self._trace > 0:
            self._trace_stack(stack, remaining_text)

    def _trace_reduce(self, stack, production, remaining_text):
        """
        Print trace output displaying that ``production`` was used to
        reduce ``stack``.

        :rtype: None
        """
        if self._trace > 2:
            rhs = ' '.join(production.rhs())
            print(f'Reduce {production.lhs()!r} <- {rhs}')
        if self._trace == 2:
            self._trace_stack(stack, remaining_text, 'R')
        elif self._trace > 1:
            self._trace_stack(stack, remaining_text)

    def _check_grammar(self):
        """
        Check to make sure that all of the CFG productions are
        potentially useful.  If any productions can never be used,
        then print a warning.

        :rtype: None
        """
        productions = self._grammar.productions()
        for i in range(len(productions)):
            for j in range(i + 1, len(productions)):
                rhs1 = productions[i].rhs()
                rhs2 = productions[j].rhs()
                if rhs1[:len(rhs2)] == rhs2:
                    print('Warning: %r will never be used' % productions[i])