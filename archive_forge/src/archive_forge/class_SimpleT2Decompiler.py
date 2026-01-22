from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
class SimpleT2Decompiler(object):

    def __init__(self, localSubrs, globalSubrs, private=None, blender=None):
        self.localSubrs = localSubrs
        self.localBias = calcSubrBias(localSubrs)
        self.globalSubrs = globalSubrs
        self.globalBias = calcSubrBias(globalSubrs)
        self.private = private
        self.blender = blender
        self.reset()

    def reset(self):
        self.callingStack = []
        self.operandStack = []
        self.hintCount = 0
        self.hintMaskBytes = 0
        self.numRegions = 0
        self.vsIndex = 0

    def execute(self, charString):
        self.callingStack.append(charString)
        needsDecompilation = charString.needsDecompilation()
        if needsDecompilation:
            program = []
            pushToProgram = program.append
        else:
            pushToProgram = lambda x: None
        pushToStack = self.operandStack.append
        index = 0
        while True:
            token, isOperator, index = charString.getToken(index)
            if token is None:
                break
            pushToProgram(token)
            if isOperator:
                handlerName = 'op_' + token
                handler = getattr(self, handlerName, None)
                if handler is not None:
                    rv = handler(index)
                    if rv:
                        hintMaskBytes, index = rv
                        pushToProgram(hintMaskBytes)
                else:
                    self.popall()
            else:
                pushToStack(token)
        if needsDecompilation:
            charString.setProgram(program)
        del self.callingStack[-1]

    def pop(self):
        value = self.operandStack[-1]
        del self.operandStack[-1]
        return value

    def popall(self):
        stack = self.operandStack[:]
        self.operandStack[:] = []
        return stack

    def push(self, value):
        self.operandStack.append(value)

    def op_return(self, index):
        if self.operandStack:
            pass

    def op_endchar(self, index):
        pass

    def op_ignore(self, index):
        pass

    def op_callsubr(self, index):
        subrIndex = self.pop()
        subr = self.localSubrs[subrIndex + self.localBias]
        self.execute(subr)

    def op_callgsubr(self, index):
        subrIndex = self.pop()
        subr = self.globalSubrs[subrIndex + self.globalBias]
        self.execute(subr)

    def op_hstem(self, index):
        self.countHints()

    def op_vstem(self, index):
        self.countHints()

    def op_hstemhm(self, index):
        self.countHints()

    def op_vstemhm(self, index):
        self.countHints()

    def op_hintmask(self, index):
        if not self.hintMaskBytes:
            self.countHints()
            self.hintMaskBytes = (self.hintCount + 7) // 8
        hintMaskBytes, index = self.callingStack[-1].getBytes(index, self.hintMaskBytes)
        return (hintMaskBytes, index)
    op_cntrmask = op_hintmask

    def countHints(self):
        args = self.popall()
        self.hintCount = self.hintCount + len(args) // 2

    def op_and(self, index):
        raise NotImplementedError

    def op_or(self, index):
        raise NotImplementedError

    def op_not(self, index):
        raise NotImplementedError

    def op_store(self, index):
        raise NotImplementedError

    def op_abs(self, index):
        raise NotImplementedError

    def op_add(self, index):
        raise NotImplementedError

    def op_sub(self, index):
        raise NotImplementedError

    def op_div(self, index):
        raise NotImplementedError

    def op_load(self, index):
        raise NotImplementedError

    def op_neg(self, index):
        raise NotImplementedError

    def op_eq(self, index):
        raise NotImplementedError

    def op_drop(self, index):
        raise NotImplementedError

    def op_put(self, index):
        raise NotImplementedError

    def op_get(self, index):
        raise NotImplementedError

    def op_ifelse(self, index):
        raise NotImplementedError

    def op_random(self, index):
        raise NotImplementedError

    def op_mul(self, index):
        raise NotImplementedError

    def op_sqrt(self, index):
        raise NotImplementedError

    def op_dup(self, index):
        raise NotImplementedError

    def op_exch(self, index):
        raise NotImplementedError

    def op_index(self, index):
        raise NotImplementedError

    def op_roll(self, index):
        raise NotImplementedError

    def op_blend(self, index):
        if self.numRegions == 0:
            self.numRegions = self.private.getNumRegions()
        numBlends = self.pop()
        numOps = numBlends * (self.numRegions + 1)
        if self.blender is None:
            del self.operandStack[-(numOps - numBlends):]
        else:
            argi = len(self.operandStack) - numOps
            end_args = tuplei = argi + numBlends
            while argi < end_args:
                next_ti = tuplei + self.numRegions
                deltas = self.operandStack[tuplei:next_ti]
                delta = self.blender(self.vsIndex, deltas)
                self.operandStack[argi] += delta
                tuplei = next_ti
                argi += 1
            self.operandStack[end_args:] = []

    def op_vsindex(self, index):
        vi = self.pop()
        self.vsIndex = vi
        self.numRegions = self.private.getNumRegions(vi)