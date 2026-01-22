from pyparsing import (Word, ZeroOrMore, printables, Suppress, OneOrMore, Group,
def __antlrAlternativesConverter(pyparsingRules, antlrBlock):
    rule = None
    if hasattr(antlrBlock, 'alternatives') and antlrBlock.alternatives != '' and (len(antlrBlock.alternatives) > 0):
        alternatives = []
        alternatives.append(__antlrAlternativeConverter(pyparsingRules, antlrBlock.a1))
        for alternative in antlrBlock.alternatives:
            alternatives.append(__antlrAlternativeConverter(pyparsingRules, alternative))
        rule = MatchFirst(alternatives)('anonymous_or')
    elif hasattr(antlrBlock, 'a1') and antlrBlock.a1 != '':
        rule = __antlrAlternativeConverter(pyparsingRules, antlrBlock.a1)
    else:
        raise Exception('Not yet implemented')
    assert rule != None
    return rule