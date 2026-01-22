from io import StringIO
from antlr4.Token import Token
from antlr4.CommonTokenStream import CommonTokenStream
def _reduceToSingleOperationPerIndex(self, rewrites):
    for i, rop in enumerate(rewrites):
        if any((rop is None, not isinstance(rop, TokenStreamRewriter.ReplaceOp))):
            continue
        inserts = [op for op in rewrites[:i] if isinstance(op, TokenStreamRewriter.InsertBeforeOp)]
        for iop in inserts:
            if iop.index == rop.index:
                rewrites[iop.instructionIndex] = None
                rop.text = '{}{}'.format(iop.text, rop.text)
            elif all((iop.index > rop.index, iop.index <= rop.last_index)):
                rewrites[iop.instructionIndex] = None
        prevReplaces = [op for op in rewrites[:i] if isinstance(op, TokenStreamRewriter.ReplaceOp)]
        for prevRop in prevReplaces:
            if all((prevRop.index >= rop.index, prevRop.last_index <= rop.last_index)):
                rewrites[prevRop.instructionIndex] = None
                continue
            isDisjoint = any((prevRop.last_index < rop.index, prevRop.index > rop.last_index))
            if all((prevRop.text is None, rop.text is None, not isDisjoint)):
                rewrites[prevRop.instructionIndex] = None
                rop.index = min(prevRop.index, rop.index)
                rop.last_index = min(prevRop.last_index, rop.last_index)
                print('New rop {}'.format(rop))
            elif not isDisjoint:
                raise ValueError('replace op boundaries of {} overlap with previous {}'.format(rop, prevRop))
    for i, iop in enumerate(rewrites):
        if any((iop is None, not isinstance(iop, TokenStreamRewriter.InsertBeforeOp))):
            continue
        prevInserts = [op for op in rewrites[:i] if isinstance(op, TokenStreamRewriter.InsertBeforeOp)]
        for prev_index, prevIop in enumerate(prevInserts):
            if prevIop.index == iop.index and type(prevIop) is TokenStreamRewriter.InsertBeforeOp:
                iop.text += prevIop.text
                rewrites[prev_index] = None
            elif prevIop.index == iop.index and type(prevIop) is TokenStreamRewriter.InsertAfterOp:
                iop.text = prevIop.text + iop.text
                rewrites[prev_index] = None
        prevReplaces = [op for op in rewrites[:i] if isinstance(op, TokenStreamRewriter.ReplaceOp)]
        for rop in prevReplaces:
            if iop.index == rop.index:
                rop.text = iop.text + rop.text
                rewrites[i] = None
                continue
            if all((iop.index >= rop.index, iop.index <= rop.last_index)):
                raise ValueError('insert op {} within boundaries of previous {}'.format(iop, rop))
    reduced = {}
    for i, op in enumerate(rewrites):
        if op is None:
            continue
        if reduced.get(op.index):
            raise ValueError('should be only one op per index')
        reduced[op.index] = op
    return reduced