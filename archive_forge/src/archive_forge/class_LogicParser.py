import operator
import re
from collections import defaultdict
from functools import reduce, total_ordering
from nltk.internals import Counter
from nltk.util import Trie
class LogicParser:
    """A lambda calculus expression parser."""

    def __init__(self, type_check=False):
        """
        :param type_check: should type checking be performed
            to their types?
        :type type_check: bool
        """
        assert isinstance(type_check, bool)
        self._currentIndex = 0
        self._buffer = []
        self.type_check = type_check
        'A list of tuples of quote characters.  The 4-tuple is comprised\n        of the start character, the end character, the escape character, and\n        a boolean indicating whether the quotes should be included in the\n        result. Quotes are used to signify that a token should be treated as\n        atomic, ignoring any special characters within the token.  The escape\n        character allows the quote end character to be used within the quote.\n        If True, the boolean indicates that the final token should contain the\n        quote and escape characters.\n        This method exists to be overridden'
        self.quote_chars = []
        self.operator_precedence = dict([(x, 1) for x in Tokens.LAMBDA_LIST] + [(x, 2) for x in Tokens.NOT_LIST] + [(APP, 3)] + [(x, 4) for x in Tokens.EQ_LIST + Tokens.NEQ_LIST] + [(x, 5) for x in Tokens.QUANTS] + [(x, 6) for x in Tokens.AND_LIST] + [(x, 7) for x in Tokens.OR_LIST] + [(x, 8) for x in Tokens.IMP_LIST] + [(x, 9) for x in Tokens.IFF_LIST] + [(None, 10)])
        self.right_associated_operations = [APP]

    def parse(self, data, signature=None):
        """
        Parse the expression.

        :param data: str for the input to be parsed
        :param signature: ``dict<str, str>`` that maps variable names to type
            strings
        :returns: a parsed Expression
        """
        data = data.rstrip()
        self._currentIndex = 0
        self._buffer, mapping = self.process(data)
        try:
            result = self.process_next_expression(None)
            if self.inRange(0):
                raise UnexpectedTokenException(self._currentIndex + 1, self.token(0))
        except LogicalExpressionException as e:
            msg = '{}\n{}\n{}^'.format(e, data, ' ' * mapping[e.index - 1])
            raise LogicalExpressionException(None, msg) from e
        if self.type_check:
            result.typecheck(signature)
        return result

    def process(self, data):
        """Split the data into tokens"""
        out = []
        mapping = {}
        tokenTrie = Trie(self.get_all_symbols())
        token = ''
        data_idx = 0
        token_start_idx = data_idx
        while data_idx < len(data):
            cur_data_idx = data_idx
            quoted_token, data_idx = self.process_quoted_token(data_idx, data)
            if quoted_token:
                if not token:
                    token_start_idx = cur_data_idx
                token += quoted_token
                continue
            st = tokenTrie
            c = data[data_idx]
            symbol = ''
            while c in st:
                symbol += c
                st = st[c]
                if len(data) - data_idx > len(symbol):
                    c = data[data_idx + len(symbol)]
                else:
                    break
            if Trie.LEAF in st:
                if token:
                    mapping[len(out)] = token_start_idx
                    out.append(token)
                    token = ''
                mapping[len(out)] = data_idx
                out.append(symbol)
                data_idx += len(symbol)
            else:
                if data[data_idx] in ' \t\n':
                    if token:
                        mapping[len(out)] = token_start_idx
                        out.append(token)
                        token = ''
                else:
                    if not token:
                        token_start_idx = data_idx
                    token += data[data_idx]
                data_idx += 1
        if token:
            mapping[len(out)] = token_start_idx
            out.append(token)
        mapping[len(out)] = len(data)
        mapping[len(out) + 1] = len(data) + 1
        return (out, mapping)

    def process_quoted_token(self, data_idx, data):
        token = ''
        c = data[data_idx]
        i = data_idx
        for start, end, escape, incl_quotes in self.quote_chars:
            if c == start:
                if incl_quotes:
                    token += c
                i += 1
                while data[i] != end:
                    if data[i] == escape:
                        if incl_quotes:
                            token += data[i]
                        i += 1
                        if len(data) == i:
                            raise LogicalExpressionException(None, 'End of input reached.  Escape character [%s] found at end.' % escape)
                        token += data[i]
                    else:
                        token += data[i]
                    i += 1
                    if len(data) == i:
                        raise LogicalExpressionException(None, 'End of input reached.  Expected: [%s]' % end)
                if incl_quotes:
                    token += data[i]
                i += 1
                if not token:
                    raise LogicalExpressionException(None, 'Empty quoted token found')
                break
        return (token, i)

    def get_all_symbols(self):
        """This method exists to be overridden"""
        return Tokens.SYMBOLS

    def inRange(self, location):
        """Return TRUE if the given location is within the buffer"""
        return self._currentIndex + location < len(self._buffer)

    def token(self, location=None):
        """Get the next waiting token.  If a location is given, then
        return the token at currentIndex+location without advancing
        currentIndex; setting it gives lookahead/lookback capability."""
        try:
            if location is None:
                tok = self._buffer[self._currentIndex]
                self._currentIndex += 1
            else:
                tok = self._buffer[self._currentIndex + location]
            return tok
        except IndexError as e:
            raise ExpectedMoreTokensException(self._currentIndex + 1) from e

    def isvariable(self, tok):
        return tok not in Tokens.TOKENS

    def process_next_expression(self, context):
        """Parse the next complete expression from the stream and return it."""
        try:
            tok = self.token()
        except ExpectedMoreTokensException as e:
            raise ExpectedMoreTokensException(self._currentIndex + 1, message='Expression expected.') from e
        accum = self.handle(tok, context)
        if not accum:
            raise UnexpectedTokenException(self._currentIndex, tok, message='Expression expected.')
        return self.attempt_adjuncts(accum, context)

    def handle(self, tok, context):
        """This method is intended to be overridden for logics that
        use different operators or expressions"""
        if self.isvariable(tok):
            return self.handle_variable(tok, context)
        elif tok in Tokens.NOT_LIST:
            return self.handle_negation(tok, context)
        elif tok in Tokens.LAMBDA_LIST:
            return self.handle_lambda(tok, context)
        elif tok in Tokens.QUANTS:
            return self.handle_quant(tok, context)
        elif tok == Tokens.OPEN:
            return self.handle_open(tok, context)

    def attempt_adjuncts(self, expression, context):
        cur_idx = None
        while cur_idx != self._currentIndex:
            cur_idx = self._currentIndex
            expression = self.attempt_EqualityExpression(expression, context)
            expression = self.attempt_ApplicationExpression(expression, context)
            expression = self.attempt_BooleanExpression(expression, context)
        return expression

    def handle_negation(self, tok, context):
        return self.make_NegatedExpression(self.process_next_expression(Tokens.NOT))

    def make_NegatedExpression(self, expression):
        return NegatedExpression(expression)

    def handle_variable(self, tok, context):
        accum = self.make_VariableExpression(tok)
        if self.inRange(0) and self.token(0) == Tokens.OPEN:
            if not isinstance(accum, FunctionVariableExpression) and (not isinstance(accum, ConstantExpression)):
                raise LogicalExpressionException(self._currentIndex, "'%s' is an illegal predicate name.  Individual variables may not be used as predicates." % tok)
            self.token()
            accum = self.make_ApplicationExpression(accum, self.process_next_expression(APP))
            while self.inRange(0) and self.token(0) == Tokens.COMMA:
                self.token()
                accum = self.make_ApplicationExpression(accum, self.process_next_expression(APP))
            self.assertNextToken(Tokens.CLOSE)
        return accum

    def get_next_token_variable(self, description):
        try:
            tok = self.token()
        except ExpectedMoreTokensException as e:
            raise ExpectedMoreTokensException(e.index, 'Variable expected.') from e
        if isinstance(self.make_VariableExpression(tok), ConstantExpression):
            raise LogicalExpressionException(self._currentIndex, "'%s' is an illegal variable name.  Constants may not be %s." % (tok, description))
        return Variable(tok)

    def handle_lambda(self, tok, context):
        if not self.inRange(0):
            raise ExpectedMoreTokensException(self._currentIndex + 2, message='Variable and Expression expected following lambda operator.')
        vars = [self.get_next_token_variable('abstracted')]
        while True:
            if not self.inRange(0) or (self.token(0) == Tokens.DOT and (not self.inRange(1))):
                raise ExpectedMoreTokensException(self._currentIndex + 2, message='Expression expected.')
            if not self.isvariable(self.token(0)):
                break
            vars.append(self.get_next_token_variable('abstracted'))
        if self.inRange(0) and self.token(0) == Tokens.DOT:
            self.token()
        accum = self.process_next_expression(tok)
        while vars:
            accum = self.make_LambdaExpression(vars.pop(), accum)
        return accum

    def handle_quant(self, tok, context):
        factory = self.get_QuantifiedExpression_factory(tok)
        if not self.inRange(0):
            raise ExpectedMoreTokensException(self._currentIndex + 2, message="Variable and Expression expected following quantifier '%s'." % tok)
        vars = [self.get_next_token_variable('quantified')]
        while True:
            if not self.inRange(0) or (self.token(0) == Tokens.DOT and (not self.inRange(1))):
                raise ExpectedMoreTokensException(self._currentIndex + 2, message='Expression expected.')
            if not self.isvariable(self.token(0)):
                break
            vars.append(self.get_next_token_variable('quantified'))
        if self.inRange(0) and self.token(0) == Tokens.DOT:
            self.token()
        accum = self.process_next_expression(tok)
        while vars:
            accum = self.make_QuanifiedExpression(factory, vars.pop(), accum)
        return accum

    def get_QuantifiedExpression_factory(self, tok):
        """This method serves as a hook for other logic parsers that
        have different quantifiers"""
        if tok in Tokens.EXISTS_LIST:
            return ExistsExpression
        elif tok in Tokens.ALL_LIST:
            return AllExpression
        elif tok in Tokens.IOTA_LIST:
            return IotaExpression
        else:
            self.assertToken(tok, Tokens.QUANTS)

    def make_QuanifiedExpression(self, factory, variable, term):
        return factory(variable, term)

    def handle_open(self, tok, context):
        accum = self.process_next_expression(None)
        self.assertNextToken(Tokens.CLOSE)
        return accum

    def attempt_EqualityExpression(self, expression, context):
        """Attempt to make an equality expression.  If the next token is an
        equality operator, then an EqualityExpression will be returned.
        Otherwise, the parameter will be returned."""
        if self.inRange(0):
            tok = self.token(0)
            if tok in Tokens.EQ_LIST + Tokens.NEQ_LIST and self.has_priority(tok, context):
                self.token()
                expression = self.make_EqualityExpression(expression, self.process_next_expression(tok))
                if tok in Tokens.NEQ_LIST:
                    expression = self.make_NegatedExpression(expression)
        return expression

    def make_EqualityExpression(self, first, second):
        """This method serves as a hook for other logic parsers that
        have different equality expression classes"""
        return EqualityExpression(first, second)

    def attempt_BooleanExpression(self, expression, context):
        """Attempt to make a boolean expression.  If the next token is a boolean
        operator, then a BooleanExpression will be returned.  Otherwise, the
        parameter will be returned."""
        while self.inRange(0):
            tok = self.token(0)
            factory = self.get_BooleanExpression_factory(tok)
            if factory and self.has_priority(tok, context):
                self.token()
                expression = self.make_BooleanExpression(factory, expression, self.process_next_expression(tok))
            else:
                break
        return expression

    def get_BooleanExpression_factory(self, tok):
        """This method serves as a hook for other logic parsers that
        have different boolean operators"""
        if tok in Tokens.AND_LIST:
            return AndExpression
        elif tok in Tokens.OR_LIST:
            return OrExpression
        elif tok in Tokens.IMP_LIST:
            return ImpExpression
        elif tok in Tokens.IFF_LIST:
            return IffExpression
        else:
            return None

    def make_BooleanExpression(self, factory, first, second):
        return factory(first, second)

    def attempt_ApplicationExpression(self, expression, context):
        """Attempt to make an application expression.  The next tokens are
        a list of arguments in parens, then the argument expression is a
        function being applied to the arguments.  Otherwise, return the
        argument expression."""
        if self.has_priority(APP, context):
            if self.inRange(0) and self.token(0) == Tokens.OPEN:
                if not isinstance(expression, LambdaExpression) and (not isinstance(expression, ApplicationExpression)) and (not isinstance(expression, FunctionVariableExpression)) and (not isinstance(expression, ConstantExpression)):
                    raise LogicalExpressionException(self._currentIndex, "The function '%s" % expression + "' is not a Lambda Expression, an Application Expression, or a functional predicate, so it may not take arguments.")
                self.token()
                accum = self.make_ApplicationExpression(expression, self.process_next_expression(APP))
                while self.inRange(0) and self.token(0) == Tokens.COMMA:
                    self.token()
                    accum = self.make_ApplicationExpression(accum, self.process_next_expression(APP))
                self.assertNextToken(Tokens.CLOSE)
                return accum
        return expression

    def make_ApplicationExpression(self, function, argument):
        return ApplicationExpression(function, argument)

    def make_VariableExpression(self, name):
        return VariableExpression(Variable(name))

    def make_LambdaExpression(self, variable, term):
        return LambdaExpression(variable, term)

    def has_priority(self, operation, context):
        return self.operator_precedence[operation] < self.operator_precedence[context] or (operation in self.right_associated_operations and self.operator_precedence[operation] == self.operator_precedence[context])

    def assertNextToken(self, expected):
        try:
            tok = self.token()
        except ExpectedMoreTokensException as e:
            raise ExpectedMoreTokensException(e.index, message="Expected token '%s'." % expected) from e
        if isinstance(expected, list):
            if tok not in expected:
                raise UnexpectedTokenException(self._currentIndex, tok, expected)
        elif tok != expected:
            raise UnexpectedTokenException(self._currentIndex, tok, expected)

    def assertToken(self, tok, expected):
        if isinstance(expected, list):
            if tok not in expected:
                raise UnexpectedTokenException(self._currentIndex, tok, expected)
        elif tok != expected:
            raise UnexpectedTokenException(self._currentIndex, tok, expected)

    def __repr__(self):
        if self.inRange(0):
            msg = 'Next token: ' + self.token(0)
        else:
            msg = 'No more tokens'
        return '<' + self.__class__.__name__ + ': ' + msg + '>'