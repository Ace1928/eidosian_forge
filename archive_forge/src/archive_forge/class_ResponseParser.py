import json, sys
from xml.dom import minidom
import plistlib
import logging
class ResponseParser(object):

    def __init__(self):
        self.meta = '*'

    def parse(self, text, pvars):
        return text

    def getMeta(self):
        return self.meta

    def getVars(self, pvars):
        if type(pvars) is dict:
            return pvars
        if type(pvars) is list:
            out = {}
            for p in pvars:
                out[p] = p
            return out