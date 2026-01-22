from antlr4 import *
from io import StringIO
import sys
class CreateTableClausesContext(ParserRuleContext):
    __slots__ = 'parser'

    def __init__(self, parser, parent: ParserRuleContext=None, invokingState: int=-1):
        super().__init__(parent, invokingState)
        self.parser = parser
        self.options = None
        self.partitioning = None
        self.tableProps = None

    def bucketSpec(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.BucketSpecContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.BucketSpecContext, i)

    def locationSpec(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.LocationSpecContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.LocationSpecContext, i)

    def commentSpec(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.CommentSpecContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.CommentSpecContext, i)

    def OPTIONS(self, i: int=None):
        if i is None:
            return self.getTokens(fugue_sqlParser.OPTIONS)
        else:
            return self.getToken(fugue_sqlParser.OPTIONS, i)

    def PARTITIONED(self, i: int=None):
        if i is None:
            return self.getTokens(fugue_sqlParser.PARTITIONED)
        else:
            return self.getToken(fugue_sqlParser.PARTITIONED, i)

    def BY(self, i: int=None):
        if i is None:
            return self.getTokens(fugue_sqlParser.BY)
        else:
            return self.getToken(fugue_sqlParser.BY, i)

    def TBLPROPERTIES(self, i: int=None):
        if i is None:
            return self.getTokens(fugue_sqlParser.TBLPROPERTIES)
        else:
            return self.getToken(fugue_sqlParser.TBLPROPERTIES, i)

    def tablePropertyList(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.TablePropertyListContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.TablePropertyListContext, i)

    def transformList(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.TransformListContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.TransformListContext, i)

    def getRuleIndex(self):
        return fugue_sqlParser.RULE_createTableClauses

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitCreateTableClauses'):
            return visitor.visitCreateTableClauses(self)
        else:
            return visitor.visitChildren(self)