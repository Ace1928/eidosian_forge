from io import StringIO
from antlr4.Token import Token
from antlr4.Utils import escapeWhitespace
from antlr4.tree.Tree import RuleNode, ErrorNode, TerminalNode, Tree, ParseTree
@classmethod
def _findAllNodes(cls, t: ParseTree, index: int, findTokens: bool, nodes: list):
    from antlr4.ParserRuleContext import ParserRuleContext
    if findTokens and isinstance(t, TerminalNode):
        if t.symbol.type == index:
            nodes.append(t)
    elif not findTokens and isinstance(t, ParserRuleContext):
        if t.ruleIndex == index:
            nodes.append(t)
    for i in range(0, t.getChildCount()):
        cls._findAllNodes(t.getChild(i), index, findTokens, nodes)