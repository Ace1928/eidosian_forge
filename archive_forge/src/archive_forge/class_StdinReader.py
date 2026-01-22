import os
import sys
import subprocess
from urllib.parse import quote
from paste.util import converters
class StdinReader:

    def __init__(self, stdin, content_length):
        self.stdin = stdin
        self.content_length = content_length

    @classmethod
    def from_environ(cls, environ):
        length = environ.get('CONTENT_LENGTH')
        if length:
            length = int(length)
        else:
            length = 0
        return cls(environ['wsgi.input'], length)

    def read(self, size=None):
        if not self.content_length:
            return b''
        if size is None:
            text = self.stdin.read(self.content_length)
        else:
            text = self.stdin.read(min(self.content_length, size))
        self.content_length -= len(text)
        return text