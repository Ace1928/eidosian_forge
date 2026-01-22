import operator
from functools import reduce
from itertools import chain
from nltk.sem.logic import (
class DrtParser(LogicParser):
    """A lambda calculus expression parser."""

    def __init__(self):
        LogicParser.__init__(self)
        self.operator_precedence = dict([(x, 1) for x in DrtTokens.LAMBDA_LIST] + [(x, 2) for x in DrtTokens.NOT_LIST] + [(APP, 3)] + [(x, 4) for x in DrtTokens.EQ_LIST + Tokens.NEQ_LIST] + [(DrtTokens.COLON, 5)] + [(DrtTokens.DRS_CONC, 6)] + [(x, 7) for x in DrtTokens.OR_LIST] + [(x, 8) for x in DrtTokens.IMP_LIST] + [(None, 9)])

    def get_all_symbols(self):
        """This method exists to be overridden"""
        return DrtTokens.SYMBOLS

    def isvariable(self, tok):
        return tok not in DrtTokens.TOKENS

    def handle(self, tok, context):
        """This method is intended to be overridden for logics that
        use different operators or expressions"""
        if tok in DrtTokens.NOT_LIST:
            return self.handle_negation(tok, context)
        elif tok in DrtTokens.LAMBDA_LIST:
            return self.handle_lambda(tok, context)
        elif tok == DrtTokens.OPEN:
            if self.inRange(0) and self.token(0) == DrtTokens.OPEN_BRACKET:
                return self.handle_DRS(tok, context)
            else:
                return self.handle_open(tok, context)
        elif tok.upper() == DrtTokens.DRS:
            self.assertNextToken(DrtTokens.OPEN)
            return self.handle_DRS(tok, context)
        elif self.isvariable(tok):
            if self.inRange(0) and self.token(0) == DrtTokens.COLON:
                return self.handle_prop(tok, context)
            else:
                return self.handle_variable(tok, context)

    def make_NegatedExpression(self, expression):
        return DrtNegatedExpression(expression)

    def handle_DRS(self, tok, context):
        refs = self.handle_refs()
        if self.inRange(0) and self.token(0) == DrtTokens.COMMA:
            self.token()
        conds = self.handle_conds(context)
        self.assertNextToken(DrtTokens.CLOSE)
        return DRS(refs, conds, None)

    def handle_refs(self):
        self.assertNextToken(DrtTokens.OPEN_BRACKET)
        refs = []
        while self.inRange(0) and self.token(0) != DrtTokens.CLOSE_BRACKET:
            if refs and self.token(0) == DrtTokens.COMMA:
                self.token()
            refs.append(self.get_next_token_variable('quantified'))
        self.assertNextToken(DrtTokens.CLOSE_BRACKET)
        return refs

    def handle_conds(self, context):
        self.assertNextToken(DrtTokens.OPEN_BRACKET)
        conds = []
        while self.inRange(0) and self.token(0) != DrtTokens.CLOSE_BRACKET:
            if conds and self.token(0) == DrtTokens.COMMA:
                self.token()
            conds.append(self.process_next_expression(context))
        self.assertNextToken(DrtTokens.CLOSE_BRACKET)
        return conds

    def handle_prop(self, tok, context):
        variable = self.make_VariableExpression(tok)
        self.assertNextToken(':')
        drs = self.process_next_expression(DrtTokens.COLON)
        return DrtProposition(variable, drs)

    def make_EqualityExpression(self, first, second):
        """This method serves as a hook for other logic parsers that
        have different equality expression classes"""
        return DrtEqualityExpression(first, second)

    def get_BooleanExpression_factory(self, tok):
        """This method serves as a hook for other logic parsers that
        have different boolean operators"""
        if tok == DrtTokens.DRS_CONC:
            return lambda first, second: DrtConcatenation(first, second, None)
        elif tok in DrtTokens.OR_LIST:
            return DrtOrExpression
        elif tok in DrtTokens.IMP_LIST:

            def make_imp_expression(first, second):
                if isinstance(first, DRS):
                    return DRS(first.refs, first.conds, second)
                if isinstance(first, DrtConcatenation):
                    return DrtConcatenation(first.first, first.second, second)
                raise Exception('Antecedent of implication must be a DRS')
            return make_imp_expression
        else:
            return None

    def make_BooleanExpression(self, factory, first, second):
        return factory(first, second)

    def make_ApplicationExpression(self, function, argument):
        return DrtApplicationExpression(function, argument)

    def make_VariableExpression(self, name):
        return DrtVariableExpression(Variable(name))

    def make_LambdaExpression(self, variables, term):
        return DrtLambdaExpression(variables, term)