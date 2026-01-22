from antlr4 import *
class sqlVisitor(ParseTreeVisitor):

    def visitSingleStatement(self, ctx: sqlParser.SingleStatementContext):
        return self.visitChildren(ctx)

    def visitSingleExpression(self, ctx: sqlParser.SingleExpressionContext):
        return self.visitChildren(ctx)

    def visitSingleTableIdentifier(self, ctx: sqlParser.SingleTableIdentifierContext):
        return self.visitChildren(ctx)

    def visitSingleMultipartIdentifier(self, ctx: sqlParser.SingleMultipartIdentifierContext):
        return self.visitChildren(ctx)

    def visitSingleFunctionIdentifier(self, ctx: sqlParser.SingleFunctionIdentifierContext):
        return self.visitChildren(ctx)

    def visitSingleDataType(self, ctx: sqlParser.SingleDataTypeContext):
        return self.visitChildren(ctx)

    def visitSingleTableSchema(self, ctx: sqlParser.SingleTableSchemaContext):
        return self.visitChildren(ctx)

    def visitStatementDefault(self, ctx: sqlParser.StatementDefaultContext):
        return self.visitChildren(ctx)

    def visitDmlStatement(self, ctx: sqlParser.DmlStatementContext):
        return self.visitChildren(ctx)

    def visitUse(self, ctx: sqlParser.UseContext):
        return self.visitChildren(ctx)

    def visitCreateNamespace(self, ctx: sqlParser.CreateNamespaceContext):
        return self.visitChildren(ctx)

    def visitSetNamespaceProperties(self, ctx: sqlParser.SetNamespacePropertiesContext):
        return self.visitChildren(ctx)

    def visitSetNamespaceLocation(self, ctx: sqlParser.SetNamespaceLocationContext):
        return self.visitChildren(ctx)

    def visitDropNamespace(self, ctx: sqlParser.DropNamespaceContext):
        return self.visitChildren(ctx)

    def visitShowNamespaces(self, ctx: sqlParser.ShowNamespacesContext):
        return self.visitChildren(ctx)

    def visitCreateTable(self, ctx: sqlParser.CreateTableContext):
        return self.visitChildren(ctx)

    def visitCreateHiveTable(self, ctx: sqlParser.CreateHiveTableContext):
        return self.visitChildren(ctx)

    def visitCreateTableLike(self, ctx: sqlParser.CreateTableLikeContext):
        return self.visitChildren(ctx)

    def visitReplaceTable(self, ctx: sqlParser.ReplaceTableContext):
        return self.visitChildren(ctx)

    def visitAnalyze(self, ctx: sqlParser.AnalyzeContext):
        return self.visitChildren(ctx)

    def visitAddTableColumns(self, ctx: sqlParser.AddTableColumnsContext):
        return self.visitChildren(ctx)

    def visitRenameTableColumn(self, ctx: sqlParser.RenameTableColumnContext):
        return self.visitChildren(ctx)

    def visitDropTableColumns(self, ctx: sqlParser.DropTableColumnsContext):
        return self.visitChildren(ctx)

    def visitRenameTable(self, ctx: sqlParser.RenameTableContext):
        return self.visitChildren(ctx)

    def visitSetTableProperties(self, ctx: sqlParser.SetTablePropertiesContext):
        return self.visitChildren(ctx)

    def visitUnsetTableProperties(self, ctx: sqlParser.UnsetTablePropertiesContext):
        return self.visitChildren(ctx)

    def visitAlterTableAlterColumn(self, ctx: sqlParser.AlterTableAlterColumnContext):
        return self.visitChildren(ctx)

    def visitHiveChangeColumn(self, ctx: sqlParser.HiveChangeColumnContext):
        return self.visitChildren(ctx)

    def visitHiveReplaceColumns(self, ctx: sqlParser.HiveReplaceColumnsContext):
        return self.visitChildren(ctx)

    def visitSetTableSerDe(self, ctx: sqlParser.SetTableSerDeContext):
        return self.visitChildren(ctx)

    def visitAddTablePartition(self, ctx: sqlParser.AddTablePartitionContext):
        return self.visitChildren(ctx)

    def visitRenameTablePartition(self, ctx: sqlParser.RenameTablePartitionContext):
        return self.visitChildren(ctx)

    def visitDropTablePartitions(self, ctx: sqlParser.DropTablePartitionsContext):
        return self.visitChildren(ctx)

    def visitSetTableLocation(self, ctx: sqlParser.SetTableLocationContext):
        return self.visitChildren(ctx)

    def visitRecoverPartitions(self, ctx: sqlParser.RecoverPartitionsContext):
        return self.visitChildren(ctx)

    def visitDropTable(self, ctx: sqlParser.DropTableContext):
        return self.visitChildren(ctx)

    def visitDropView(self, ctx: sqlParser.DropViewContext):
        return self.visitChildren(ctx)

    def visitCreateView(self, ctx: sqlParser.CreateViewContext):
        return self.visitChildren(ctx)

    def visitCreateTempViewUsing(self, ctx: sqlParser.CreateTempViewUsingContext):
        return self.visitChildren(ctx)

    def visitAlterViewQuery(self, ctx: sqlParser.AlterViewQueryContext):
        return self.visitChildren(ctx)

    def visitCreateFunction(self, ctx: sqlParser.CreateFunctionContext):
        return self.visitChildren(ctx)

    def visitDropFunction(self, ctx: sqlParser.DropFunctionContext):
        return self.visitChildren(ctx)

    def visitExplain(self, ctx: sqlParser.ExplainContext):
        return self.visitChildren(ctx)

    def visitShowTables(self, ctx: sqlParser.ShowTablesContext):
        return self.visitChildren(ctx)

    def visitShowTable(self, ctx: sqlParser.ShowTableContext):
        return self.visitChildren(ctx)

    def visitShowTblProperties(self, ctx: sqlParser.ShowTblPropertiesContext):
        return self.visitChildren(ctx)

    def visitShowColumns(self, ctx: sqlParser.ShowColumnsContext):
        return self.visitChildren(ctx)

    def visitShowViews(self, ctx: sqlParser.ShowViewsContext):
        return self.visitChildren(ctx)

    def visitShowPartitions(self, ctx: sqlParser.ShowPartitionsContext):
        return self.visitChildren(ctx)

    def visitShowFunctions(self, ctx: sqlParser.ShowFunctionsContext):
        return self.visitChildren(ctx)

    def visitShowCreateTable(self, ctx: sqlParser.ShowCreateTableContext):
        return self.visitChildren(ctx)

    def visitShowCurrentNamespace(self, ctx: sqlParser.ShowCurrentNamespaceContext):
        return self.visitChildren(ctx)

    def visitDescribeFunction(self, ctx: sqlParser.DescribeFunctionContext):
        return self.visitChildren(ctx)

    def visitDescribeNamespace(self, ctx: sqlParser.DescribeNamespaceContext):
        return self.visitChildren(ctx)

    def visitDescribeRelation(self, ctx: sqlParser.DescribeRelationContext):
        return self.visitChildren(ctx)

    def visitDescribeQuery(self, ctx: sqlParser.DescribeQueryContext):
        return self.visitChildren(ctx)

    def visitCommentNamespace(self, ctx: sqlParser.CommentNamespaceContext):
        return self.visitChildren(ctx)

    def visitCommentTable(self, ctx: sqlParser.CommentTableContext):
        return self.visitChildren(ctx)

    def visitRefreshTable(self, ctx: sqlParser.RefreshTableContext):
        return self.visitChildren(ctx)

    def visitRefreshResource(self, ctx: sqlParser.RefreshResourceContext):
        return self.visitChildren(ctx)

    def visitCacheTable(self, ctx: sqlParser.CacheTableContext):
        return self.visitChildren(ctx)

    def visitUncacheTable(self, ctx: sqlParser.UncacheTableContext):
        return self.visitChildren(ctx)

    def visitClearCache(self, ctx: sqlParser.ClearCacheContext):
        return self.visitChildren(ctx)

    def visitLoadData(self, ctx: sqlParser.LoadDataContext):
        return self.visitChildren(ctx)

    def visitTruncateTable(self, ctx: sqlParser.TruncateTableContext):
        return self.visitChildren(ctx)

    def visitRepairTable(self, ctx: sqlParser.RepairTableContext):
        return self.visitChildren(ctx)

    def visitManageResource(self, ctx: sqlParser.ManageResourceContext):
        return self.visitChildren(ctx)

    def visitFailNativeCommand(self, ctx: sqlParser.FailNativeCommandContext):
        return self.visitChildren(ctx)

    def visitSetConfiguration(self, ctx: sqlParser.SetConfigurationContext):
        return self.visitChildren(ctx)

    def visitResetConfiguration(self, ctx: sqlParser.ResetConfigurationContext):
        return self.visitChildren(ctx)

    def visitUnsupportedHiveNativeCommands(self, ctx: sqlParser.UnsupportedHiveNativeCommandsContext):
        return self.visitChildren(ctx)

    def visitCreateTableHeader(self, ctx: sqlParser.CreateTableHeaderContext):
        return self.visitChildren(ctx)

    def visitReplaceTableHeader(self, ctx: sqlParser.ReplaceTableHeaderContext):
        return self.visitChildren(ctx)

    def visitBucketSpec(self, ctx: sqlParser.BucketSpecContext):
        return self.visitChildren(ctx)

    def visitSkewSpec(self, ctx: sqlParser.SkewSpecContext):
        return self.visitChildren(ctx)

    def visitLocationSpec(self, ctx: sqlParser.LocationSpecContext):
        return self.visitChildren(ctx)

    def visitCommentSpec(self, ctx: sqlParser.CommentSpecContext):
        return self.visitChildren(ctx)

    def visitQuery(self, ctx: sqlParser.QueryContext):
        return self.visitChildren(ctx)

    def visitInsertOverwriteTable(self, ctx: sqlParser.InsertOverwriteTableContext):
        return self.visitChildren(ctx)

    def visitInsertIntoTable(self, ctx: sqlParser.InsertIntoTableContext):
        return self.visitChildren(ctx)

    def visitInsertOverwriteHiveDir(self, ctx: sqlParser.InsertOverwriteHiveDirContext):
        return self.visitChildren(ctx)

    def visitInsertOverwriteDir(self, ctx: sqlParser.InsertOverwriteDirContext):
        return self.visitChildren(ctx)

    def visitPartitionSpecLocation(self, ctx: sqlParser.PartitionSpecLocationContext):
        return self.visitChildren(ctx)

    def visitPartitionSpec(self, ctx: sqlParser.PartitionSpecContext):
        return self.visitChildren(ctx)

    def visitPartitionVal(self, ctx: sqlParser.PartitionValContext):
        return self.visitChildren(ctx)

    def visitNamespace(self, ctx: sqlParser.NamespaceContext):
        return self.visitChildren(ctx)

    def visitDescribeFuncName(self, ctx: sqlParser.DescribeFuncNameContext):
        return self.visitChildren(ctx)

    def visitDescribeColName(self, ctx: sqlParser.DescribeColNameContext):
        return self.visitChildren(ctx)

    def visitCtes(self, ctx: sqlParser.CtesContext):
        return self.visitChildren(ctx)

    def visitNamedQuery(self, ctx: sqlParser.NamedQueryContext):
        return self.visitChildren(ctx)

    def visitTableProvider(self, ctx: sqlParser.TableProviderContext):
        return self.visitChildren(ctx)

    def visitCreateTableClauses(self, ctx: sqlParser.CreateTableClausesContext):
        return self.visitChildren(ctx)

    def visitTablePropertyList(self, ctx: sqlParser.TablePropertyListContext):
        return self.visitChildren(ctx)

    def visitTableProperty(self, ctx: sqlParser.TablePropertyContext):
        return self.visitChildren(ctx)

    def visitTablePropertyKey(self, ctx: sqlParser.TablePropertyKeyContext):
        return self.visitChildren(ctx)

    def visitTablePropertyValue(self, ctx: sqlParser.TablePropertyValueContext):
        return self.visitChildren(ctx)

    def visitConstantList(self, ctx: sqlParser.ConstantListContext):
        return self.visitChildren(ctx)

    def visitNestedConstantList(self, ctx: sqlParser.NestedConstantListContext):
        return self.visitChildren(ctx)

    def visitCreateFileFormat(self, ctx: sqlParser.CreateFileFormatContext):
        return self.visitChildren(ctx)

    def visitTableFileFormat(self, ctx: sqlParser.TableFileFormatContext):
        return self.visitChildren(ctx)

    def visitGenericFileFormat(self, ctx: sqlParser.GenericFileFormatContext):
        return self.visitChildren(ctx)

    def visitStorageHandler(self, ctx: sqlParser.StorageHandlerContext):
        return self.visitChildren(ctx)

    def visitResource(self, ctx: sqlParser.ResourceContext):
        return self.visitChildren(ctx)

    def visitSingleInsertQuery(self, ctx: sqlParser.SingleInsertQueryContext):
        return self.visitChildren(ctx)

    def visitMultiInsertQuery(self, ctx: sqlParser.MultiInsertQueryContext):
        return self.visitChildren(ctx)

    def visitDeleteFromTable(self, ctx: sqlParser.DeleteFromTableContext):
        return self.visitChildren(ctx)

    def visitUpdateTable(self, ctx: sqlParser.UpdateTableContext):
        return self.visitChildren(ctx)

    def visitMergeIntoTable(self, ctx: sqlParser.MergeIntoTableContext):
        return self.visitChildren(ctx)

    def visitQueryOrganization(self, ctx: sqlParser.QueryOrganizationContext):
        return self.visitChildren(ctx)

    def visitMultiInsertQueryBody(self, ctx: sqlParser.MultiInsertQueryBodyContext):
        return self.visitChildren(ctx)

    def visitQueryTermDefault(self, ctx: sqlParser.QueryTermDefaultContext):
        return self.visitChildren(ctx)

    def visitSetOperation(self, ctx: sqlParser.SetOperationContext):
        return self.visitChildren(ctx)

    def visitQueryPrimaryDefault(self, ctx: sqlParser.QueryPrimaryDefaultContext):
        return self.visitChildren(ctx)

    def visitFromStmt(self, ctx: sqlParser.FromStmtContext):
        return self.visitChildren(ctx)

    def visitTable(self, ctx: sqlParser.TableContext):
        return self.visitChildren(ctx)

    def visitInlineTableDefault1(self, ctx: sqlParser.InlineTableDefault1Context):
        return self.visitChildren(ctx)

    def visitSubquery(self, ctx: sqlParser.SubqueryContext):
        return self.visitChildren(ctx)

    def visitSortItem(self, ctx: sqlParser.SortItemContext):
        return self.visitChildren(ctx)

    def visitFromStatement(self, ctx: sqlParser.FromStatementContext):
        return self.visitChildren(ctx)

    def visitFromStatementBody(self, ctx: sqlParser.FromStatementBodyContext):
        return self.visitChildren(ctx)

    def visitTransformQuerySpecification(self, ctx: sqlParser.TransformQuerySpecificationContext):
        return self.visitChildren(ctx)

    def visitRegularQuerySpecification(self, ctx: sqlParser.RegularQuerySpecificationContext):
        return self.visitChildren(ctx)

    def visitTransformClause(self, ctx: sqlParser.TransformClauseContext):
        return self.visitChildren(ctx)

    def visitSelectClause(self, ctx: sqlParser.SelectClauseContext):
        return self.visitChildren(ctx)

    def visitSetClause(self, ctx: sqlParser.SetClauseContext):
        return self.visitChildren(ctx)

    def visitMatchedClause(self, ctx: sqlParser.MatchedClauseContext):
        return self.visitChildren(ctx)

    def visitNotMatchedClause(self, ctx: sqlParser.NotMatchedClauseContext):
        return self.visitChildren(ctx)

    def visitMatchedAction(self, ctx: sqlParser.MatchedActionContext):
        return self.visitChildren(ctx)

    def visitNotMatchedAction(self, ctx: sqlParser.NotMatchedActionContext):
        return self.visitChildren(ctx)

    def visitAssignmentList(self, ctx: sqlParser.AssignmentListContext):
        return self.visitChildren(ctx)

    def visitAssignment(self, ctx: sqlParser.AssignmentContext):
        return self.visitChildren(ctx)

    def visitWhereClause(self, ctx: sqlParser.WhereClauseContext):
        return self.visitChildren(ctx)

    def visitHavingClause(self, ctx: sqlParser.HavingClauseContext):
        return self.visitChildren(ctx)

    def visitHint(self, ctx: sqlParser.HintContext):
        return self.visitChildren(ctx)

    def visitHintStatement(self, ctx: sqlParser.HintStatementContext):
        return self.visitChildren(ctx)

    def visitFromClause(self, ctx: sqlParser.FromClauseContext):
        return self.visitChildren(ctx)

    def visitAggregationClause(self, ctx: sqlParser.AggregationClauseContext):
        return self.visitChildren(ctx)

    def visitGroupingSet(self, ctx: sqlParser.GroupingSetContext):
        return self.visitChildren(ctx)

    def visitPivotClause(self, ctx: sqlParser.PivotClauseContext):
        return self.visitChildren(ctx)

    def visitPivotColumn(self, ctx: sqlParser.PivotColumnContext):
        return self.visitChildren(ctx)

    def visitPivotValue(self, ctx: sqlParser.PivotValueContext):
        return self.visitChildren(ctx)

    def visitLateralView(self, ctx: sqlParser.LateralViewContext):
        return self.visitChildren(ctx)

    def visitSetQuantifier(self, ctx: sqlParser.SetQuantifierContext):
        return self.visitChildren(ctx)

    def visitRelation(self, ctx: sqlParser.RelationContext):
        return self.visitChildren(ctx)

    def visitJoinRelation(self, ctx: sqlParser.JoinRelationContext):
        return self.visitChildren(ctx)

    def visitJoinType(self, ctx: sqlParser.JoinTypeContext):
        return self.visitChildren(ctx)

    def visitJoinCriteria(self, ctx: sqlParser.JoinCriteriaContext):
        return self.visitChildren(ctx)

    def visitSample(self, ctx: sqlParser.SampleContext):
        return self.visitChildren(ctx)

    def visitSampleByPercentile(self, ctx: sqlParser.SampleByPercentileContext):
        return self.visitChildren(ctx)

    def visitSampleByRows(self, ctx: sqlParser.SampleByRowsContext):
        return self.visitChildren(ctx)

    def visitSampleByBucket(self, ctx: sqlParser.SampleByBucketContext):
        return self.visitChildren(ctx)

    def visitSampleByBytes(self, ctx: sqlParser.SampleByBytesContext):
        return self.visitChildren(ctx)

    def visitIdentifierList(self, ctx: sqlParser.IdentifierListContext):
        return self.visitChildren(ctx)

    def visitIdentifierSeq(self, ctx: sqlParser.IdentifierSeqContext):
        return self.visitChildren(ctx)

    def visitOrderedIdentifierList(self, ctx: sqlParser.OrderedIdentifierListContext):
        return self.visitChildren(ctx)

    def visitOrderedIdentifier(self, ctx: sqlParser.OrderedIdentifierContext):
        return self.visitChildren(ctx)

    def visitIdentifierCommentList(self, ctx: sqlParser.IdentifierCommentListContext):
        return self.visitChildren(ctx)

    def visitIdentifierComment(self, ctx: sqlParser.IdentifierCommentContext):
        return self.visitChildren(ctx)

    def visitTableName(self, ctx: sqlParser.TableNameContext):
        return self.visitChildren(ctx)

    def visitAliasedQuery(self, ctx: sqlParser.AliasedQueryContext):
        return self.visitChildren(ctx)

    def visitAliasedRelation(self, ctx: sqlParser.AliasedRelationContext):
        return self.visitChildren(ctx)

    def visitInlineTable(self, ctx: sqlParser.InlineTableContext):
        return self.visitChildren(ctx)

    def visitFunctionTable(self, ctx: sqlParser.FunctionTableContext):
        return self.visitChildren(ctx)

    def visitTableAlias(self, ctx: sqlParser.TableAliasContext):
        return self.visitChildren(ctx)

    def visitRowFormatSerde(self, ctx: sqlParser.RowFormatSerdeContext):
        return self.visitChildren(ctx)

    def visitRowFormatDelimited(self, ctx: sqlParser.RowFormatDelimitedContext):
        return self.visitChildren(ctx)

    def visitMultipartIdentifierList(self, ctx: sqlParser.MultipartIdentifierListContext):
        return self.visitChildren(ctx)

    def visitMultipartIdentifier(self, ctx: sqlParser.MultipartIdentifierContext):
        return self.visitChildren(ctx)

    def visitTableIdentifier(self, ctx: sqlParser.TableIdentifierContext):
        return self.visitChildren(ctx)

    def visitFunctionIdentifier(self, ctx: sqlParser.FunctionIdentifierContext):
        return self.visitChildren(ctx)

    def visitNamedExpression(self, ctx: sqlParser.NamedExpressionContext):
        return self.visitChildren(ctx)

    def visitNamedExpressionSeq(self, ctx: sqlParser.NamedExpressionSeqContext):
        return self.visitChildren(ctx)

    def visitTransformList(self, ctx: sqlParser.TransformListContext):
        return self.visitChildren(ctx)

    def visitIdentityTransform(self, ctx: sqlParser.IdentityTransformContext):
        return self.visitChildren(ctx)

    def visitApplyTransform(self, ctx: sqlParser.ApplyTransformContext):
        return self.visitChildren(ctx)

    def visitTransformArgument(self, ctx: sqlParser.TransformArgumentContext):
        return self.visitChildren(ctx)

    def visitExpression(self, ctx: sqlParser.ExpressionContext):
        return self.visitChildren(ctx)

    def visitLogicalNot(self, ctx: sqlParser.LogicalNotContext):
        return self.visitChildren(ctx)

    def visitPredicated(self, ctx: sqlParser.PredicatedContext):
        return self.visitChildren(ctx)

    def visitExists(self, ctx: sqlParser.ExistsContext):
        return self.visitChildren(ctx)

    def visitLogicalBinary(self, ctx: sqlParser.LogicalBinaryContext):
        return self.visitChildren(ctx)

    def visitPredicate(self, ctx: sqlParser.PredicateContext):
        return self.visitChildren(ctx)

    def visitValueExpressionDefault(self, ctx: sqlParser.ValueExpressionDefaultContext):
        return self.visitChildren(ctx)

    def visitComparison(self, ctx: sqlParser.ComparisonContext):
        return self.visitChildren(ctx)

    def visitArithmeticBinary(self, ctx: sqlParser.ArithmeticBinaryContext):
        return self.visitChildren(ctx)

    def visitArithmeticUnary(self, ctx: sqlParser.ArithmeticUnaryContext):
        return self.visitChildren(ctx)

    def visitStruct(self, ctx: sqlParser.StructContext):
        return self.visitChildren(ctx)

    def visitDereference(self, ctx: sqlParser.DereferenceContext):
        return self.visitChildren(ctx)

    def visitSimpleCase(self, ctx: sqlParser.SimpleCaseContext):
        return self.visitChildren(ctx)

    def visitColumnReference(self, ctx: sqlParser.ColumnReferenceContext):
        return self.visitChildren(ctx)

    def visitRowConstructor(self, ctx: sqlParser.RowConstructorContext):
        return self.visitChildren(ctx)

    def visitLast(self, ctx: sqlParser.LastContext):
        return self.visitChildren(ctx)

    def visitStar(self, ctx: sqlParser.StarContext):
        return self.visitChildren(ctx)

    def visitOverlay(self, ctx: sqlParser.OverlayContext):
        return self.visitChildren(ctx)

    def visitSubscript(self, ctx: sqlParser.SubscriptContext):
        return self.visitChildren(ctx)

    def visitSubqueryExpression(self, ctx: sqlParser.SubqueryExpressionContext):
        return self.visitChildren(ctx)

    def visitSubstring(self, ctx: sqlParser.SubstringContext):
        return self.visitChildren(ctx)

    def visitCurrentDatetime(self, ctx: sqlParser.CurrentDatetimeContext):
        return self.visitChildren(ctx)

    def visitCast(self, ctx: sqlParser.CastContext):
        return self.visitChildren(ctx)

    def visitConstantDefault(self, ctx: sqlParser.ConstantDefaultContext):
        return self.visitChildren(ctx)

    def visitLambda(self, ctx: sqlParser.LambdaContext):
        return self.visitChildren(ctx)

    def visitParenthesizedExpression(self, ctx: sqlParser.ParenthesizedExpressionContext):
        return self.visitChildren(ctx)

    def visitExtract(self, ctx: sqlParser.ExtractContext):
        return self.visitChildren(ctx)

    def visitTrim(self, ctx: sqlParser.TrimContext):
        return self.visitChildren(ctx)

    def visitFunctionCall(self, ctx: sqlParser.FunctionCallContext):
        return self.visitChildren(ctx)

    def visitSearchedCase(self, ctx: sqlParser.SearchedCaseContext):
        return self.visitChildren(ctx)

    def visitPosition(self, ctx: sqlParser.PositionContext):
        return self.visitChildren(ctx)

    def visitFirst(self, ctx: sqlParser.FirstContext):
        return self.visitChildren(ctx)

    def visitNullLiteral(self, ctx: sqlParser.NullLiteralContext):
        return self.visitChildren(ctx)

    def visitIntervalLiteral(self, ctx: sqlParser.IntervalLiteralContext):
        return self.visitChildren(ctx)

    def visitTypeConstructor(self, ctx: sqlParser.TypeConstructorContext):
        return self.visitChildren(ctx)

    def visitNumericLiteral(self, ctx: sqlParser.NumericLiteralContext):
        return self.visitChildren(ctx)

    def visitBooleanLiteral(self, ctx: sqlParser.BooleanLiteralContext):
        return self.visitChildren(ctx)

    def visitStringLiteral(self, ctx: sqlParser.StringLiteralContext):
        return self.visitChildren(ctx)

    def visitComparisonOperator(self, ctx: sqlParser.ComparisonOperatorContext):
        return self.visitChildren(ctx)

    def visitArithmeticOperator(self, ctx: sqlParser.ArithmeticOperatorContext):
        return self.visitChildren(ctx)

    def visitPredicateOperator(self, ctx: sqlParser.PredicateOperatorContext):
        return self.visitChildren(ctx)

    def visitBooleanValue(self, ctx: sqlParser.BooleanValueContext):
        return self.visitChildren(ctx)

    def visitInterval(self, ctx: sqlParser.IntervalContext):
        return self.visitChildren(ctx)

    def visitErrorCapturingMultiUnitsInterval(self, ctx: sqlParser.ErrorCapturingMultiUnitsIntervalContext):
        return self.visitChildren(ctx)

    def visitMultiUnitsInterval(self, ctx: sqlParser.MultiUnitsIntervalContext):
        return self.visitChildren(ctx)

    def visitErrorCapturingUnitToUnitInterval(self, ctx: sqlParser.ErrorCapturingUnitToUnitIntervalContext):
        return self.visitChildren(ctx)

    def visitUnitToUnitInterval(self, ctx: sqlParser.UnitToUnitIntervalContext):
        return self.visitChildren(ctx)

    def visitIntervalValue(self, ctx: sqlParser.IntervalValueContext):
        return self.visitChildren(ctx)

    def visitIntervalUnit(self, ctx: sqlParser.IntervalUnitContext):
        return self.visitChildren(ctx)

    def visitColPosition(self, ctx: sqlParser.ColPositionContext):
        return self.visitChildren(ctx)

    def visitComplexDataType(self, ctx: sqlParser.ComplexDataTypeContext):
        return self.visitChildren(ctx)

    def visitPrimitiveDataType(self, ctx: sqlParser.PrimitiveDataTypeContext):
        return self.visitChildren(ctx)

    def visitQualifiedColTypeWithPositionList(self, ctx: sqlParser.QualifiedColTypeWithPositionListContext):
        return self.visitChildren(ctx)

    def visitQualifiedColTypeWithPosition(self, ctx: sqlParser.QualifiedColTypeWithPositionContext):
        return self.visitChildren(ctx)

    def visitColTypeList(self, ctx: sqlParser.ColTypeListContext):
        return self.visitChildren(ctx)

    def visitColType(self, ctx: sqlParser.ColTypeContext):
        return self.visitChildren(ctx)

    def visitComplexColTypeList(self, ctx: sqlParser.ComplexColTypeListContext):
        return self.visitChildren(ctx)

    def visitComplexColType(self, ctx: sqlParser.ComplexColTypeContext):
        return self.visitChildren(ctx)

    def visitWhenClause(self, ctx: sqlParser.WhenClauseContext):
        return self.visitChildren(ctx)

    def visitWindowClause(self, ctx: sqlParser.WindowClauseContext):
        return self.visitChildren(ctx)

    def visitNamedWindow(self, ctx: sqlParser.NamedWindowContext):
        return self.visitChildren(ctx)

    def visitWindowRef(self, ctx: sqlParser.WindowRefContext):
        return self.visitChildren(ctx)

    def visitWindowDef(self, ctx: sqlParser.WindowDefContext):
        return self.visitChildren(ctx)

    def visitWindowFrame(self, ctx: sqlParser.WindowFrameContext):
        return self.visitChildren(ctx)

    def visitFrameBound(self, ctx: sqlParser.FrameBoundContext):
        return self.visitChildren(ctx)

    def visitQualifiedNameList(self, ctx: sqlParser.QualifiedNameListContext):
        return self.visitChildren(ctx)

    def visitFunctionName(self, ctx: sqlParser.FunctionNameContext):
        return self.visitChildren(ctx)

    def visitQualifiedName(self, ctx: sqlParser.QualifiedNameContext):
        return self.visitChildren(ctx)

    def visitErrorCapturingIdentifier(self, ctx: sqlParser.ErrorCapturingIdentifierContext):
        return self.visitChildren(ctx)

    def visitErrorIdent(self, ctx: sqlParser.ErrorIdentContext):
        return self.visitChildren(ctx)

    def visitRealIdent(self, ctx: sqlParser.RealIdentContext):
        return self.visitChildren(ctx)

    def visitIdentifier(self, ctx: sqlParser.IdentifierContext):
        return self.visitChildren(ctx)

    def visitUnquotedIdentifier(self, ctx: sqlParser.UnquotedIdentifierContext):
        return self.visitChildren(ctx)

    def visitQuotedIdentifierAlternative(self, ctx: sqlParser.QuotedIdentifierAlternativeContext):
        return self.visitChildren(ctx)

    def visitQuotedIdentifier(self, ctx: sqlParser.QuotedIdentifierContext):
        return self.visitChildren(ctx)

    def visitExponentLiteral(self, ctx: sqlParser.ExponentLiteralContext):
        return self.visitChildren(ctx)

    def visitDecimalLiteral(self, ctx: sqlParser.DecimalLiteralContext):
        return self.visitChildren(ctx)

    def visitLegacyDecimalLiteral(self, ctx: sqlParser.LegacyDecimalLiteralContext):
        return self.visitChildren(ctx)

    def visitIntegerLiteral(self, ctx: sqlParser.IntegerLiteralContext):
        return self.visitChildren(ctx)

    def visitBigIntLiteral(self, ctx: sqlParser.BigIntLiteralContext):
        return self.visitChildren(ctx)

    def visitSmallIntLiteral(self, ctx: sqlParser.SmallIntLiteralContext):
        return self.visitChildren(ctx)

    def visitTinyIntLiteral(self, ctx: sqlParser.TinyIntLiteralContext):
        return self.visitChildren(ctx)

    def visitDoubleLiteral(self, ctx: sqlParser.DoubleLiteralContext):
        return self.visitChildren(ctx)

    def visitBigDecimalLiteral(self, ctx: sqlParser.BigDecimalLiteralContext):
        return self.visitChildren(ctx)

    def visitAlterColumnAction(self, ctx: sqlParser.AlterColumnActionContext):
        return self.visitChildren(ctx)

    def visitAnsiNonReserved(self, ctx: sqlParser.AnsiNonReservedContext):
        return self.visitChildren(ctx)

    def visitStrictNonReserved(self, ctx: sqlParser.StrictNonReservedContext):
        return self.visitChildren(ctx)

    def visitNonReserved(self, ctx: sqlParser.NonReservedContext):
        return self.visitChildren(ctx)