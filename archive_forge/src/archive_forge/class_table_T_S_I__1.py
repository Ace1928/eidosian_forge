from . import DefaultTable
from fontTools.misc.loggingTools import LogMixin
from fontTools.misc.textTools import strjoin, tobytes, tostr
class table_T_S_I__1(LogMixin, DefaultTable.DefaultTable):
    extras = {65530: 'ppgm', 65531: 'cvt', 65532: 'reserved', 65533: 'fpgm'}
    indextable = 'TSI0'

    def decompile(self, data, ttFont):
        totalLength = len(data)
        indextable = ttFont[self.indextable]
        for indices, isExtra in zip((indextable.indices, indextable.extra_indices), (False, True)):
            programs = {}
            for i, (glyphID, textLength, textOffset) in enumerate(indices):
                if isExtra:
                    name = self.extras[glyphID]
                else:
                    name = ttFont.getGlyphName(glyphID)
                if textOffset > totalLength:
                    self.log.warning('textOffset > totalLength; %r skipped' % name)
                    continue
                if textLength < 32768:
                    pass
                elif textLength == 32768:
                    isLast = i == len(indices) - 1
                    if isLast:
                        if isExtra:
                            nextTextOffset = totalLength
                        else:
                            nextTextOffset = indextable.extra_indices[0][2]
                    else:
                        nextTextOffset = indices[i + 1][2]
                    assert nextTextOffset >= textOffset, 'entries not sorted by offset'
                    if nextTextOffset > totalLength:
                        self.log.warning('nextTextOffset > totalLength; %r truncated' % name)
                        nextTextOffset = totalLength
                    textLength = nextTextOffset - textOffset
                else:
                    from fontTools import ttLib
                    raise ttLib.TTLibError('%r textLength (%d) must not be > 32768' % (name, textLength))
                text = data[textOffset:textOffset + textLength]
                assert len(text) == textLength
                text = tostr(text, encoding='utf-8')
                if text:
                    programs[name] = text
            if isExtra:
                self.extraPrograms = programs
            else:
                self.glyphPrograms = programs

    def compile(self, ttFont):
        if not hasattr(self, 'glyphPrograms'):
            self.glyphPrograms = {}
            self.extraPrograms = {}
        data = b''
        indextable = ttFont[self.indextable]
        glyphNames = ttFont.getGlyphOrder()
        indices = []
        for i in range(len(glyphNames)):
            if len(data) % 2:
                data = data + b'\r'
            name = glyphNames[i]
            if name in self.glyphPrograms:
                text = tobytes(self.glyphPrograms[name], encoding='utf-8')
            else:
                text = b''
            textLength = len(text)
            if textLength >= 32768:
                textLength = 32768
            indices.append((i, textLength, len(data)))
            data = data + text
        extra_indices = []
        codes = sorted(self.extras.items())
        for i in range(len(codes)):
            if len(data) % 2:
                data = data + b'\r'
            code, name = codes[i]
            if name in self.extraPrograms:
                text = tobytes(self.extraPrograms[name], encoding='utf-8')
            else:
                text = b''
            textLength = len(text)
            if textLength >= 32768:
                textLength = 32768
            extra_indices.append((code, textLength, len(data)))
            data = data + text
        indextable.set(indices, extra_indices)
        return data

    def toXML(self, writer, ttFont):
        names = sorted(self.glyphPrograms.keys())
        writer.newline()
        for name in names:
            text = self.glyphPrograms[name]
            if not text:
                continue
            writer.begintag('glyphProgram', name=name)
            writer.newline()
            writer.write_noindent(text.replace('\r', '\n'))
            writer.newline()
            writer.endtag('glyphProgram')
            writer.newline()
            writer.newline()
        extra_names = sorted(self.extraPrograms.keys())
        for name in extra_names:
            text = self.extraPrograms[name]
            if not text:
                continue
            writer.begintag('extraProgram', name=name)
            writer.newline()
            writer.write_noindent(text.replace('\r', '\n'))
            writer.newline()
            writer.endtag('extraProgram')
            writer.newline()
            writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        if not hasattr(self, 'glyphPrograms'):
            self.glyphPrograms = {}
            self.extraPrograms = {}
        lines = strjoin(content).replace('\r', '\n').split('\n')
        text = '\r'.join(lines[1:-1])
        if name == 'glyphProgram':
            self.glyphPrograms[attrs['name']] = text
        elif name == 'extraProgram':
            self.extraPrograms[attrs['name']] = text