from fontTools.misc.textTools import bytesjoin, strjoin, tobytes, tostr, safeEval
from fontTools.misc import sstruct
from . import DefaultTable
import base64
class table_D_S_I_G_(DefaultTable.DefaultTable):

    def decompile(self, data, ttFont):
        dummy, newData = sstruct.unpack2(DSIG_HeaderFormat, data, self)
        assert self.ulVersion == 1, 'DSIG ulVersion must be 1'
        assert self.usFlag & ~1 == 0, 'DSIG usFlag must be 0x1 or 0x0'
        self.signatureRecords = sigrecs = []
        for n in range(self.usNumSigs):
            sigrec, newData = sstruct.unpack2(DSIG_SignatureFormat, newData, SignatureRecord())
            assert sigrec.ulFormat == 1, 'DSIG signature record #%d ulFormat must be 1' % n
            sigrecs.append(sigrec)
        for sigrec in sigrecs:
            dummy, newData = sstruct.unpack2(DSIG_SignatureBlockFormat, data[sigrec.ulOffset:], sigrec)
            assert sigrec.usReserved1 == 0, 'DSIG signature record #%d usReserverd1 must be 0' % n
            assert sigrec.usReserved2 == 0, 'DSIG signature record #%d usReserverd2 must be 0' % n
            sigrec.pkcs7 = newData[:sigrec.cbSignature]

    def compile(self, ttFont):
        packed = sstruct.pack(DSIG_HeaderFormat, self)
        headers = [packed]
        offset = len(packed) + self.usNumSigs * sstruct.calcsize(DSIG_SignatureFormat)
        data = []
        for sigrec in self.signatureRecords:
            sigrec.cbSignature = len(sigrec.pkcs7)
            packed = sstruct.pack(DSIG_SignatureBlockFormat, sigrec) + sigrec.pkcs7
            data.append(packed)
            sigrec.ulLength = len(packed)
            sigrec.ulOffset = offset
            headers.append(sstruct.pack(DSIG_SignatureFormat, sigrec))
            offset += sigrec.ulLength
        if offset % 2:
            data.append(b'\x00')
        return bytesjoin(headers + data)

    def toXML(self, xmlWriter, ttFont):
        xmlWriter.comment('note that the Digital Signature will be invalid after recompilation!')
        xmlWriter.newline()
        xmlWriter.simpletag('tableHeader', version=self.ulVersion, numSigs=self.usNumSigs, flag='0x%X' % self.usFlag)
        for sigrec in self.signatureRecords:
            xmlWriter.newline()
            sigrec.toXML(xmlWriter, ttFont)
        xmlWriter.newline()

    def fromXML(self, name, attrs, content, ttFont):
        if name == 'tableHeader':
            self.signatureRecords = []
            self.ulVersion = safeEval(attrs['version'])
            self.usNumSigs = safeEval(attrs['numSigs'])
            self.usFlag = safeEval(attrs['flag'])
            return
        if name == 'SignatureRecord':
            sigrec = SignatureRecord()
            sigrec.fromXML(name, attrs, content, ttFont)
            self.signatureRecords.append(sigrec)