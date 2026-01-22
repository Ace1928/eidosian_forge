import os
import coverage
from kivy.lang.parser import Parser
class KivyFileTracer(coverage.plugin.FileTracer):
    filename = ''

    def __init__(self, filename, **kwargs):
        super(KivyFileTracer, self).__init__(**kwargs)
        self.filename = filename

    def source_filename(self):
        return self.filename