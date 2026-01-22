from fontTools.misc.textTools import strjoin, tobytes, tostr
from . import DefaultTable
class asciiTable(DefaultTable.DefaultTable):

    def toXML(self, writer, ttFont):
        data = tostr(self.data)
        data = data.split('\x00')
        data = strjoin(data)
        writer.begintag('source')
        writer.newline()
        writer.write_noindent(data)
        writer.newline()
        writer.endtag('source')
        writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        lines = strjoin(content).split('\n')
        self.data = tobytes('\n'.join(lines[1:-1]))