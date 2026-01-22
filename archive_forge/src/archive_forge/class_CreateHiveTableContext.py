from antlr4 import *
from io import StringIO
import sys
class CreateHiveTableContext(StatementContext):

    def __init__(self, parser, ctx: ParserRuleContext):
        super().__init__(parser)
        self.columns = None
        self.partitionColumns = None
        self.partitionColumnNames = None
        self.tableProps = None
        self.copyFrom(ctx)

    def createTableHeader(self):
        return self.getTypedRuleContext(fugue_sqlParser.CreateTableHeaderContext, 0)

    def commentSpec(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.CommentSpecContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.CommentSpecContext, i)

    def bucketSpec(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.BucketSpecContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.BucketSpecContext, i)

    def skewSpec(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.SkewSpecContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.SkewSpecContext, i)

    def rowFormat(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.RowFormatContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.RowFormatContext, i)

    def createFileFormat(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.CreateFileFormatContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.CreateFileFormatContext, i)

    def locationSpec(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.LocationSpecContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.LocationSpecContext, i)

    def query(self):
        return self.getTypedRuleContext(fugue_sqlParser.QueryContext, 0)

    def colTypeList(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.ColTypeListContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.ColTypeListContext, i)

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

    def identifierList(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.IdentifierListContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.IdentifierListContext, i)

    def tablePropertyList(self, i: int=None):
        if i is None:
            return self.getTypedRuleContexts(fugue_sqlParser.TablePropertyListContext)
        else:
            return self.getTypedRuleContext(fugue_sqlParser.TablePropertyListContext, i)

    def AS(self):
        return self.getToken(fugue_sqlParser.AS, 0)

    def accept(self, visitor: ParseTreeVisitor):
        if hasattr(visitor, 'visitCreateHiveTable'):
            return visitor.visitCreateHiveTable(self)
        else:
            return visitor.visitChildren(self)