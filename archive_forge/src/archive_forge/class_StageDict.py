import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class StageDict(object):
    """A dictionary of stages corresponding to classes"""

    def __init__(self, classes, postprocessor):
        """Instantiate an element from elyxer.each class and store as a dictionary"""
        instances = self.instantiate(classes, postprocessor)
        self.stagedict = dict([(x.processedclass, x) for x in instances])

    def instantiate(self, classes, postprocessor):
        """Instantiate an element from elyxer.each class"""
        stages = [x.__new__(x) for x in classes]
        for element in stages:
            element.__init__()
            element.postprocessor = postprocessor
        return stages

    def getstage(self, element):
        """Get the stage for a given element, if the type is in the dict"""
        if not element.__class__ in self.stagedict:
            return None
        return self.stagedict[element.__class__]