from antlr4 import *
class AutolevListener(ParseTreeListener):

    def enterProg(self, ctx: AutolevParser.ProgContext):
        pass

    def exitProg(self, ctx: AutolevParser.ProgContext):
        pass

    def enterStat(self, ctx: AutolevParser.StatContext):
        pass

    def exitStat(self, ctx: AutolevParser.StatContext):
        pass

    def enterVecAssign(self, ctx: AutolevParser.VecAssignContext):
        pass

    def exitVecAssign(self, ctx: AutolevParser.VecAssignContext):
        pass

    def enterIndexAssign(self, ctx: AutolevParser.IndexAssignContext):
        pass

    def exitIndexAssign(self, ctx: AutolevParser.IndexAssignContext):
        pass

    def enterRegularAssign(self, ctx: AutolevParser.RegularAssignContext):
        pass

    def exitRegularAssign(self, ctx: AutolevParser.RegularAssignContext):
        pass

    def enterEquals(self, ctx: AutolevParser.EqualsContext):
        pass

    def exitEquals(self, ctx: AutolevParser.EqualsContext):
        pass

    def enterIndex(self, ctx: AutolevParser.IndexContext):
        pass

    def exitIndex(self, ctx: AutolevParser.IndexContext):
        pass

    def enterDiff(self, ctx: AutolevParser.DiffContext):
        pass

    def exitDiff(self, ctx: AutolevParser.DiffContext):
        pass

    def enterFunctionCall(self, ctx: AutolevParser.FunctionCallContext):
        pass

    def exitFunctionCall(self, ctx: AutolevParser.FunctionCallContext):
        pass

    def enterVarDecl(self, ctx: AutolevParser.VarDeclContext):
        pass

    def exitVarDecl(self, ctx: AutolevParser.VarDeclContext):
        pass

    def enterVarType(self, ctx: AutolevParser.VarTypeContext):
        pass

    def exitVarType(self, ctx: AutolevParser.VarTypeContext):
        pass

    def enterVarDecl2(self, ctx: AutolevParser.VarDecl2Context):
        pass

    def exitVarDecl2(self, ctx: AutolevParser.VarDecl2Context):
        pass

    def enterRanges(self, ctx: AutolevParser.RangesContext):
        pass

    def exitRanges(self, ctx: AutolevParser.RangesContext):
        pass

    def enterMassDecl(self, ctx: AutolevParser.MassDeclContext):
        pass

    def exitMassDecl(self, ctx: AutolevParser.MassDeclContext):
        pass

    def enterMassDecl2(self, ctx: AutolevParser.MassDecl2Context):
        pass

    def exitMassDecl2(self, ctx: AutolevParser.MassDecl2Context):
        pass

    def enterInertiaDecl(self, ctx: AutolevParser.InertiaDeclContext):
        pass

    def exitInertiaDecl(self, ctx: AutolevParser.InertiaDeclContext):
        pass

    def enterMatrix(self, ctx: AutolevParser.MatrixContext):
        pass

    def exitMatrix(self, ctx: AutolevParser.MatrixContext):
        pass

    def enterMatrixInOutput(self, ctx: AutolevParser.MatrixInOutputContext):
        pass

    def exitMatrixInOutput(self, ctx: AutolevParser.MatrixInOutputContext):
        pass

    def enterCodeCommands(self, ctx: AutolevParser.CodeCommandsContext):
        pass

    def exitCodeCommands(self, ctx: AutolevParser.CodeCommandsContext):
        pass

    def enterSettings(self, ctx: AutolevParser.SettingsContext):
        pass

    def exitSettings(self, ctx: AutolevParser.SettingsContext):
        pass

    def enterUnits(self, ctx: AutolevParser.UnitsContext):
        pass

    def exitUnits(self, ctx: AutolevParser.UnitsContext):
        pass

    def enterInputs(self, ctx: AutolevParser.InputsContext):
        pass

    def exitInputs(self, ctx: AutolevParser.InputsContext):
        pass

    def enterId_diff(self, ctx: AutolevParser.Id_diffContext):
        pass

    def exitId_diff(self, ctx: AutolevParser.Id_diffContext):
        pass

    def enterInputs2(self, ctx: AutolevParser.Inputs2Context):
        pass

    def exitInputs2(self, ctx: AutolevParser.Inputs2Context):
        pass

    def enterOutputs(self, ctx: AutolevParser.OutputsContext):
        pass

    def exitOutputs(self, ctx: AutolevParser.OutputsContext):
        pass

    def enterOutputs2(self, ctx: AutolevParser.Outputs2Context):
        pass

    def exitOutputs2(self, ctx: AutolevParser.Outputs2Context):
        pass

    def enterCodegen(self, ctx: AutolevParser.CodegenContext):
        pass

    def exitCodegen(self, ctx: AutolevParser.CodegenContext):
        pass

    def enterCommands(self, ctx: AutolevParser.CommandsContext):
        pass

    def exitCommands(self, ctx: AutolevParser.CommandsContext):
        pass

    def enterVec(self, ctx: AutolevParser.VecContext):
        pass

    def exitVec(self, ctx: AutolevParser.VecContext):
        pass

    def enterParens(self, ctx: AutolevParser.ParensContext):
        pass

    def exitParens(self, ctx: AutolevParser.ParensContext):
        pass

    def enterVectorOrDyadic(self, ctx: AutolevParser.VectorOrDyadicContext):
        pass

    def exitVectorOrDyadic(self, ctx: AutolevParser.VectorOrDyadicContext):
        pass

    def enterExponent(self, ctx: AutolevParser.ExponentContext):
        pass

    def exitExponent(self, ctx: AutolevParser.ExponentContext):
        pass

    def enterMulDiv(self, ctx: AutolevParser.MulDivContext):
        pass

    def exitMulDiv(self, ctx: AutolevParser.MulDivContext):
        pass

    def enterAddSub(self, ctx: AutolevParser.AddSubContext):
        pass

    def exitAddSub(self, ctx: AutolevParser.AddSubContext):
        pass

    def enterFloat(self, ctx: AutolevParser.FloatContext):
        pass

    def exitFloat(self, ctx: AutolevParser.FloatContext):
        pass

    def enterInt(self, ctx: AutolevParser.IntContext):
        pass

    def exitInt(self, ctx: AutolevParser.IntContext):
        pass

    def enterIdEqualsExpr(self, ctx: AutolevParser.IdEqualsExprContext):
        pass

    def exitIdEqualsExpr(self, ctx: AutolevParser.IdEqualsExprContext):
        pass

    def enterNegativeOne(self, ctx: AutolevParser.NegativeOneContext):
        pass

    def exitNegativeOne(self, ctx: AutolevParser.NegativeOneContext):
        pass

    def enterFunction(self, ctx: AutolevParser.FunctionContext):
        pass

    def exitFunction(self, ctx: AutolevParser.FunctionContext):
        pass

    def enterRangess(self, ctx: AutolevParser.RangessContext):
        pass

    def exitRangess(self, ctx: AutolevParser.RangessContext):
        pass

    def enterColon(self, ctx: AutolevParser.ColonContext):
        pass

    def exitColon(self, ctx: AutolevParser.ColonContext):
        pass

    def enterId(self, ctx: AutolevParser.IdContext):
        pass

    def exitId(self, ctx: AutolevParser.IdContext):
        pass

    def enterExp(self, ctx: AutolevParser.ExpContext):
        pass

    def exitExp(self, ctx: AutolevParser.ExpContext):
        pass

    def enterMatrices(self, ctx: AutolevParser.MatricesContext):
        pass

    def exitMatrices(self, ctx: AutolevParser.MatricesContext):
        pass

    def enterIndexing(self, ctx: AutolevParser.IndexingContext):
        pass

    def exitIndexing(self, ctx: AutolevParser.IndexingContext):
        pass