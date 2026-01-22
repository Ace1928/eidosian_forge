from pyparsing import Literal,Word,Group,\
import math
import operator
def BNF():
    """
    expop   :: '^'
    multop  :: '*' | '/'
    addop   :: '+' | '-'
    integer :: ['+' | '-'] '0'..'9'+
    atom    :: PI | E | real | fn '(' expr ')' | '(' expr ')'
    factor  :: atom [ expop factor ]*
    term    :: factor [ multop factor ]*
    expr    :: term [ addop term ]*
    """
    global bnf
    if not bnf:
        point = Literal('.')
        e = CaselessKeyword('E')
        pi = CaselessKeyword('PI')
        fnumber = Regex('[+-]?\\d+(?:\\.\\d*)?(?:[eE][+-]?\\d+)?')
        ident = Word(alphas, alphanums + '_$')
        plus, minus, mult, div = map(Literal, '+-*/')
        lpar, rpar = map(Suppress, '()')
        addop = plus | minus
        multop = mult | div
        expop = Literal('^')
        expr = Forward()
        atom = ((0, None) * minus + (pi | e | fnumber | ident + lpar + expr + rpar | ident).setParseAction(pushFirst) | Group(lpar + expr + rpar)).setParseAction(pushUMinus)
        factor = Forward()
        factor << atom + ZeroOrMore((expop + factor).setParseAction(pushFirst))
        term = factor + ZeroOrMore((multop + factor).setParseAction(pushFirst))
        expr << term + ZeroOrMore((addop + term).setParseAction(pushFirst))
        bnf = expr
    return bnf