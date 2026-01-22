import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class ParameterFunction(CommandBit):
    """A function with a variable number of parameters defined in a template."""
    'The parameters are defined as a parameter definition.'

    def readparams(self, readtemplate, pos):
        """Read the params according to the template."""
        self.params = dict()
        for paramdef in self.paramdefs(readtemplate):
            paramdef.read(pos, self)
            self.params['$' + paramdef.name] = paramdef

    def paramdefs(self, readtemplate):
        """Read each param definition in the template"""
        pos = TextPosition(readtemplate)
        while not pos.finished():
            paramdef = ParameterDefinition().parse(pos)
            if paramdef:
                yield paramdef

    def getparam(self, name):
        """Get a parameter as parsed."""
        if not name in self.params:
            return None
        return self.params[name]

    def getvalue(self, name):
        """Get the value of a parameter."""
        return self.getparam(name).value

    def getliteralvalue(self, name):
        """Get the literal value of a parameter."""
        param = self.getparam(name)
        if not param or not param.literalvalue:
            return None
        return param.literalvalue