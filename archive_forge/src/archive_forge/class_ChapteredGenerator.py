import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class ChapteredGenerator(NumberGenerator):
    """Generate chaptered numbers, as in Chapter.Number."""
    'Used in equations, figures: Equation (5.3), figure 8.15.'

    def generate(self, type):
        """Generate a number which goes with first-level numbers (chapters). """
        'For the article classes a unique number is generated.'
        if DocumentParameters.startinglevel > 0:
            return NumberGenerator.generator.generate(type)
        chapter = self.getcounter('Chapter')
        return self.getdependentcounter(type, chapter).getnext()