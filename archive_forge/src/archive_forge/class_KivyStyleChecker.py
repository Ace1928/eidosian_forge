import sys
from os import walk
from os.path import isdir, join, normpath
import pep8
class KivyStyleChecker(pep8.Checker):

    def __init__(self, filename):
        pep8.Checker.__init__(self, filename, ignore=pep8_ignores)

    def report_error(self, line_number, offset, text, check):
        return pep8.Checker.report_error(self, line_number, offset, text, check)