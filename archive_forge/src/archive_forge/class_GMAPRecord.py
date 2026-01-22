from fontTools.misc import sstruct
from fontTools.misc.textTools import tobytes, tostr, safeEval
from . import DefaultTable
class GMAPRecord(object):

    def __init__(self, uv=0, cid=0, gid=0, ggid=0, name=''):
        self.UV = uv
        self.cid = cid
        self.gid = gid
        self.ggid = ggid
        self.name = name

    def toXML(self, writer, ttFont):
        writer.begintag('GMAPRecord')
        writer.newline()
        writer.simpletag('UV', value=self.UV)
        writer.newline()
        writer.simpletag('cid', value=self.cid)
        writer.newline()
        writer.simpletag('gid', value=self.gid)
        writer.newline()
        writer.simpletag('glyphletGid', value=self.gid)
        writer.newline()
        writer.simpletag('GlyphletName', value=self.name)
        writer.newline()
        writer.endtag('GMAPRecord')
        writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        value = attrs['value']
        if name == 'GlyphletName':
            self.name = value
        else:
            setattr(self, name, safeEval(value))

    def compile(self, ttFont):
        if self.UV is None:
            self.UV = 0
        nameLen = len(self.name)
        if nameLen < 32:
            self.name = self.name + '\x00' * (32 - nameLen)
        data = sstruct.pack(GMAPRecordFormat1, self)
        return data

    def __repr__(self):
        return 'GMAPRecord[ UV: ' + str(self.UV) + ', cid: ' + str(self.cid) + ', gid: ' + str(self.gid) + ', ggid: ' + str(self.ggid) + ', Glyphlet Name: ' + str(self.name) + ' ]'