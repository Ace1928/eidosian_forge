import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class FormulaParser(Parser):
    """Parses a formula"""

    def parseheader(self, reader):
        """See if the formula is inlined"""
        self.begin = reader.linenumber + 1
        type = self.parsetype(reader)
        if not type:
            reader.nextline()
            type = self.parsetype(reader)
            if not type:
                Trace.error('Unknown formula type in ' + reader.currentline().strip())
                return ['unknown']
        return [type]

    def parsetype(self, reader):
        """Get the formula type from the first line."""
        if reader.currentline().find(FormulaConfig.starts['simple']) >= 0:
            return 'inline'
        if reader.currentline().find(FormulaConfig.starts['complex']) >= 0:
            return 'block'
        if reader.currentline().find(FormulaConfig.starts['unnumbered']) >= 0:
            return 'block'
        if reader.currentline().find(FormulaConfig.starts['beginbefore']) >= 0:
            return 'numbered'
        return None

    def parse(self, reader):
        """Parse the formula until the end"""
        formula = self.parseformula(reader)
        while not reader.currentline().startswith(self.ending):
            stripped = reader.currentline().strip()
            if len(stripped) > 0:
                Trace.error('Unparsed formula line ' + stripped)
            reader.nextline()
        reader.nextline()
        return formula

    def parseformula(self, reader):
        """Parse the formula contents"""
        simple = FormulaConfig.starts['simple']
        if simple in reader.currentline():
            rest = reader.currentline().split(simple, 1)[1]
            if simple in rest:
                return self.parsesingleliner(reader, simple, simple)
            return self.parsemultiliner(reader, simple, simple)
        if FormulaConfig.starts['complex'] in reader.currentline():
            return self.parsemultiliner(reader, FormulaConfig.starts['complex'], FormulaConfig.endings['complex'])
        beginbefore = FormulaConfig.starts['beginbefore']
        beginafter = FormulaConfig.starts['beginafter']
        if beginbefore in reader.currentline():
            if reader.currentline().strip().endswith(beginafter):
                current = reader.currentline().strip()
                endsplit = current.split(beginbefore)[1].split(beginafter)
                startpiece = beginbefore + endsplit[0] + beginafter
                endbefore = FormulaConfig.endings['endbefore']
                endafter = FormulaConfig.endings['endafter']
                endpiece = endbefore + endsplit[0] + endafter
                return startpiece + self.parsemultiliner(reader, startpiece, endpiece) + endpiece
            Trace.error('Missing ' + beginafter + ' in ' + reader.currentline())
            return ''
        begincommand = FormulaConfig.starts['command']
        beginbracket = FormulaConfig.starts['bracket']
        if begincommand in reader.currentline() and beginbracket in reader.currentline():
            endbracket = FormulaConfig.endings['bracket']
            return self.parsemultiliner(reader, beginbracket, endbracket)
        Trace.error('Formula beginning ' + reader.currentline() + ' is unknown')
        return ''

    def parsesingleliner(self, reader, start, ending):
        """Parse a formula in one line"""
        line = reader.currentline().strip()
        if not start in line:
            Trace.error('Line ' + line + ' does not contain formula start ' + start)
            return ''
        if not line.endswith(ending):
            Trace.error('Formula ' + line + ' does not end with ' + ending)
            return ''
        index = line.index(start)
        rest = line[index + len(start):-len(ending)]
        reader.nextline()
        return rest

    def parsemultiliner(self, reader, start, ending):
        """Parse a formula in multiple lines"""
        formula = ''
        line = reader.currentline()
        if not start in line:
            Trace.error('Line ' + line.strip() + ' does not contain formula start ' + start)
            return ''
        index = line.index(start)
        line = line[index + len(start):].strip()
        while not line.endswith(ending):
            formula += line + '\n'
            reader.nextline()
            line = reader.currentline()
        formula += line[:-len(ending)]
        reader.nextline()
        return formula