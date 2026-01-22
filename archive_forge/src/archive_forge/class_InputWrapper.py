import re
import sys
import warnings
class InputWrapper:

    def __init__(self, wsgi_input):
        self.input = wsgi_input

    def read(self, *args):
        assert_(len(args) == 1)
        v = self.input.read(*args)
        assert_(type(v) is bytes)
        return v

    def readline(self, *args):
        assert_(len(args) <= 1)
        v = self.input.readline(*args)
        assert_(type(v) is bytes)
        return v

    def readlines(self, *args):
        assert_(len(args) <= 1)
        lines = self.input.readlines(*args)
        assert_(type(lines) is list)
        for line in lines:
            assert_(type(line) is bytes)
        return lines

    def __iter__(self):
        while 1:
            line = self.readline()
            if not line:
                return
            yield line

    def close(self):
        assert_(0, 'input.close() must not be called')