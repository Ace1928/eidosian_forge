from io import StringIO
import re
import cgi
from paste.util import threadedprint
from paste import wsgilib
from paste import response
import sys
class TeeFile(object):

    def __init__(self, files):
        self.files = files

    def write(self, v):
        for file in self.files:
            file.write(v)