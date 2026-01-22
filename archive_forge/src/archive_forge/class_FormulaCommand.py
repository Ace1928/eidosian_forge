import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class FormulaCommand(FormulaBit):
    """A LaTeX command inside a formula"""
    types = []
    start = FormulaConfig.starts['command']
    commandmap = None

    def detect(self, pos):
        """Find the current command."""
        return pos.checkfor(FormulaCommand.start)

    def parsebit(self, pos):
        """Parse the command."""
        command = self.extractcommand(pos)
        bit = self.parsewithcommand(command, pos)
        if bit:
            return bit
        if command.startswith('\\up') or command.startswith('\\Up'):
            upgreek = self.parseupgreek(command, pos)
            if upgreek:
                return upgreek
        if not self.factory.defining:
            Trace.error('Unknown command ' + command)
        self.output = TaggedOutput().settag('span class="unknown"')
        self.add(FormulaConstant(command))
        return None

    def parsewithcommand(self, command, pos):
        """Parse the command type once we have the command."""
        for type in FormulaCommand.types:
            if command in type.commandmap:
                return self.parsecommandtype(command, type, pos)
        return None

    def parsecommandtype(self, command, type, pos):
        """Parse a given command type."""
        bit = self.factory.create(type)
        bit.setcommand(command)
        returned = bit.parsebit(pos)
        if returned:
            return returned
        return bit

    def extractcommand(self, pos):
        """Extract the command from elyxer.the current position."""
        if not pos.checkskip(FormulaCommand.start):
            pos.error('Missing command start ' + FormulaCommand.start)
            return
        if pos.finished():
            return self.emptycommand(pos)
        if pos.current().isalpha():
            command = FormulaCommand.start + pos.globalpha()
            pos.checkskip('*')
            return command
        return FormulaCommand.start + pos.skipcurrent()

    def emptycommand(self, pos):
        """Check for an empty command: look for command disguised as ending.
    Special case against '{ \\{ \\} }' situation."""
        command = ''
        if not pos.isout():
            ending = pos.nextending()
            if ending and pos.checkskip(ending):
                command = ending
        return FormulaCommand.start + command

    def parseupgreek(self, command, pos):
        """Parse the Greek \\up command.."""
        if len(command) < 4:
            return None
        if command.startswith('\\up'):
            upcommand = '\\' + command[3:]
        elif pos.checkskip('\\Up'):
            upcommand = '\\' + command[3:4].upper() + command[4:]
        else:
            Trace.error('Impossible upgreek command: ' + command)
            return
        upgreek = self.parsewithcommand(upcommand, pos)
        if upgreek:
            upgreek.type = 'font'
        return upgreek