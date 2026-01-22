from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
class T2CharString(object):
    operandEncoding = t2OperandEncoding
    operators, opcodes = buildOperatorDict(t2Operators)
    decompilerClass = SimpleT2Decompiler
    outlineExtractor = T2OutlineExtractor

    def __init__(self, bytecode=None, program=None, private=None, globalSubrs=None):
        if program is None:
            program = []
        self.bytecode = bytecode
        self.program = program
        self.private = private
        self.globalSubrs = globalSubrs if globalSubrs is not None else []
        self._cur_vsindex = None

    def getNumRegions(self, vsindex=None):
        pd = self.private
        assert pd is not None
        if vsindex is not None:
            self._cur_vsindex = vsindex
        elif self._cur_vsindex is None:
            self._cur_vsindex = pd.vsindex if hasattr(pd, 'vsindex') else 0
        return pd.getNumRegions(self._cur_vsindex)

    def __repr__(self):
        if self.bytecode is None:
            return '<%s (source) at %x>' % (self.__class__.__name__, id(self))
        else:
            return '<%s (bytecode) at %x>' % (self.__class__.__name__, id(self))

    def getIntEncoder(self):
        return encodeIntT2

    def getFixedEncoder(self):
        return encodeFixed

    def decompile(self):
        if not self.needsDecompilation():
            return
        subrs = getattr(self.private, 'Subrs', [])
        decompiler = self.decompilerClass(subrs, self.globalSubrs, self.private)
        decompiler.execute(self)

    def draw(self, pen, blender=None):
        subrs = getattr(self.private, 'Subrs', [])
        extractor = self.outlineExtractor(pen, subrs, self.globalSubrs, self.private.nominalWidthX, self.private.defaultWidthX, self.private, blender)
        extractor.execute(self)
        self.width = extractor.width

    def calcBounds(self, glyphSet):
        boundsPen = BoundsPen(glyphSet)
        self.draw(boundsPen)
        return boundsPen.bounds

    def compile(self, isCFF2=False):
        if self.bytecode is not None:
            return
        opcodes = self.opcodes
        program = self.program
        if isCFF2:
            if program and program[-1] in ('return', 'endchar'):
                program = program[:-1]
        elif program and (not isinstance(program[-1], str)):
            raise CharStringCompileError('T2CharString or Subr has items on the stack after last operator.')
        bytecode = []
        encodeInt = self.getIntEncoder()
        encodeFixed = self.getFixedEncoder()
        i = 0
        end = len(program)
        while i < end:
            token = program[i]
            i = i + 1
            if isinstance(token, str):
                try:
                    bytecode.extend((bytechr(b) for b in opcodes[token]))
                except KeyError:
                    raise CharStringCompileError('illegal operator: %s' % token)
                if token in ('hintmask', 'cntrmask'):
                    bytecode.append(program[i])
                    i = i + 1
            elif isinstance(token, int):
                bytecode.append(encodeInt(token))
            elif isinstance(token, float):
                bytecode.append(encodeFixed(token))
            else:
                assert 0, 'unsupported type: %s' % type(token)
        try:
            bytecode = bytesjoin(bytecode)
        except TypeError:
            log.error(bytecode)
            raise
        self.setBytecode(bytecode)

    def needsDecompilation(self):
        return self.bytecode is not None

    def setProgram(self, program):
        self.program = program
        self.bytecode = None

    def setBytecode(self, bytecode):
        self.bytecode = bytecode
        self.program = None

    def getToken(self, index, len=len, byteord=byteord, isinstance=isinstance):
        if self.bytecode is not None:
            if index >= len(self.bytecode):
                return (None, 0, 0)
            b0 = byteord(self.bytecode[index])
            index = index + 1
            handler = self.operandEncoding[b0]
            token, index = handler(self, b0, self.bytecode, index)
        else:
            if index >= len(self.program):
                return (None, 0, 0)
            token = self.program[index]
            index = index + 1
        isOperator = isinstance(token, str)
        return (token, isOperator, index)

    def getBytes(self, index, nBytes):
        if self.bytecode is not None:
            newIndex = index + nBytes
            bytes = self.bytecode[index:newIndex]
            index = newIndex
        else:
            bytes = self.program[index]
            index = index + 1
        assert len(bytes) == nBytes
        return (bytes, index)

    def handle_operator(self, operator):
        return operator

    def toXML(self, xmlWriter, ttFont=None):
        from fontTools.misc.textTools import num2binary
        if self.bytecode is not None:
            xmlWriter.dumphex(self.bytecode)
        else:
            index = 0
            args = []
            while True:
                token, isOperator, index = self.getToken(index)
                if token is None:
                    break
                if isOperator:
                    if token in ('hintmask', 'cntrmask'):
                        hintMask, isOperator, index = self.getToken(index)
                        bits = []
                        for byte in hintMask:
                            bits.append(num2binary(byteord(byte), 8))
                        hintMask = strjoin(bits)
                        line = ' '.join(args + [token, hintMask])
                    else:
                        line = ' '.join(args + [token])
                    xmlWriter.write(line)
                    xmlWriter.newline()
                    args = []
                else:
                    if isinstance(token, float):
                        token = floatToFixedToStr(token, precisionBits=16)
                    else:
                        token = str(token)
                    args.append(token)
            if args:
                line = ' '.join(args)
                xmlWriter.write(line)

    def fromXML(self, name, attrs, content):
        from fontTools.misc.textTools import binary2num, readHex
        if attrs.get('raw'):
            self.setBytecode(readHex(content))
            return
        content = strjoin(content)
        content = content.split()
        program = []
        end = len(content)
        i = 0
        while i < end:
            token = content[i]
            i = i + 1
            try:
                token = int(token)
            except ValueError:
                try:
                    token = strToFixedToFloat(token, precisionBits=16)
                except ValueError:
                    program.append(token)
                    if token in ('hintmask', 'cntrmask'):
                        mask = content[i]
                        maskBytes = b''
                        for j in range(0, len(mask), 8):
                            maskBytes = maskBytes + bytechr(binary2num(mask[j:j + 8]))
                        program.append(maskBytes)
                        i = i + 1
                else:
                    program.append(token)
            else:
                program.append(token)
        self.setProgram(program)