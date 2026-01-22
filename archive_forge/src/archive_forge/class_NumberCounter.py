import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class NumberCounter(object):
    """A counter for numbers (by default)."""
    'The type can be changed to return letters, roman numbers...'
    name = None
    value = None
    mode = None
    master = None
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    symbols = NumberingConfig.sequence['symbols']
    romannumerals = [('M', 1000), ('CM', 900), ('D', 500), ('CD', 400), ('C', 100), ('XC', 90), ('L', 50), ('XL', 40), ('X', 10), ('IX', 9), ('V', 5), ('IV', 4), ('I', 1)]

    def __init__(self, name):
        """Give a name to the counter."""
        self.name = name

    def setmode(self, mode):
        """Set the counter mode. Can be changed at runtime."""
        self.mode = mode
        return self

    def init(self, value):
        """Set an initial value."""
        self.value = value

    def gettext(self):
        """Get the next value as a text string."""
        return str(self.value)

    def getletter(self):
        """Get the next value as a letter."""
        return self.getsequence(self.letters)

    def getsymbol(self):
        """Get the next value as a symbol."""
        return self.getsequence(self.symbols)

    def getsequence(self, sequence):
        """Get the next value from elyxer.a sequence."""
        return sequence[(self.value - 1) % len(sequence)]

    def getroman(self):
        """Get the next value as a roman number."""
        result = ''
        number = self.value
        for numeral, value in self.romannumerals:
            if number >= value:
                result += numeral * (number / value)
                number = number % value
        return result

    def getvalue(self):
        """Get the current value as configured in the current mode."""
        if not self.mode or self.mode in ['text', '1']:
            return self.gettext()
        if self.mode == 'A':
            return self.getletter()
        if self.mode == 'a':
            return self.getletter().lower()
        if self.mode == 'I':
            return self.getroman()
        if self.mode == '*':
            return self.getsymbol()
        Trace.error('Unknown counter mode ' + self.mode)
        return self.gettext()

    def getnext(self):
        """Increase the current value and get the next value as configured."""
        if not self.value:
            self.value = 0
        self.value += 1
        return self.getvalue()

    def reset(self):
        """Reset the counter."""
        self.value = 0

    def __unicode__(self):
        """Return a printable representation."""
        result = 'Counter ' + self.name
        if self.mode:
            result += ' in mode ' + self.mode
        return result