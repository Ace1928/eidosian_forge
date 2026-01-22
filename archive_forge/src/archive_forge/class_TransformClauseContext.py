from antlr4 import *
from io import StringIO
import sys
class TransformClauseContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.kind = None
        self.inRowFormat = None
        self.recordWriter = None
        self.script = None
        self.outRowFormat = None
        self.recordReader = None

    def USING(self):
        return self.getToken(fugue_sqlParser.USING, 0)

    def STRING(self, i: int=None):
        if i is None:
            return self.getTokens(fugue_sqlParser.STRING)
        else:
            return self.getToken(fugue_sqlParser.STRING, i)

    def SELECT(self):
        return self.getToken(fugue_sqlParser.SELECT, 0)

    def namedExpressionSeq(self):
        return self.getTypedRuleContext(fugue_sqlParser.NamedExpressionSeqContext, 0)

    def TRANSFORM(self):
        return self.getToken(fugue_sqlParser.TRANSFORM, 0)

    def MAP(self):
        return self.getToken(fugue_sqlParser.MAP, 0)

    def REDUCE(self):
        return self.getToken(fugue_sqlParser.REDUCE, 0)

    def RECORDWRITER(self):
        return self.getToken(fugue_sqlParser.RECORDWRITER, 0)

    def AS(self):
        return self.getToken(fugue_sqlParser.AS, 0)

    def RECORDREADER(self):
        return self.getToken(fugue_sqlParser.RECORDREADER, 0)

    def rowFormat(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.RowFormatContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.RowFormatContext, i)

    def identifierSeq(self):
        return self.getTypedRuleContext(fugue_sqlParser.IdentifierSeqContext, 0)

    def colTypeList(self):
        return self.getTypedRuleContext(fugue_sqlParser.ColTypeListContext, 0)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_transformClause

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitTransformClause'):
            return visitor.visitTransformClause(self)
        else:
            return visitor.visitChildren(self)