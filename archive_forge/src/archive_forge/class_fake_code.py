import io
import sys
import time
import marshal
class fake_code:

    def __init__(self, filename, line, name):
        self.co_filename = filename
        self.co_line = line
        self.co_name = name
        self.co_firstlineno = 0

    def __repr__(self):
        return repr((self.co_filename, self.co_line, self.co_name))