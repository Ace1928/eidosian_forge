import sys
import os
import os.path
import re
import itertools
import warnings
import unicodedata
from docutils import ApplicationError, DataError, __version_info__
from docutils import nodes
from docutils.nodes import unescape
import docutils.io
from docutils.utils.error_reporting import ErrorOutput, SafeString
class DependencyList(object):
    """
    List of dependencies, with file recording support.

    Note that the output file is not automatically closed.  You have
    to explicitly call the close() method.
    """

    def __init__(self, output_file=None, dependencies=[]):
        """
        Initialize the dependency list, automatically setting the
        output file to `output_file` (see `set_output()`) and adding
        all supplied dependencies.
        """
        self.set_output(output_file)
        for i in dependencies:
            self.add(i)

    def set_output(self, output_file):
        """
        Set the output file and clear the list of already added
        dependencies.

        `output_file` must be a string.  The specified file is
        immediately overwritten.

        If output_file is '-', the output will be written to stdout.
        If it is None, no file output is done when calling add().
        """
        self.list = []
        if output_file:
            if output_file == '-':
                of = None
            else:
                of = output_file
            self.file = docutils.io.FileOutput(destination_path=of, encoding='utf8', autoclose=False)
        else:
            self.file = None

    def add(self, *filenames):
        """
        If the dependency `filename` has not already been added,
        append it to self.list and print it to self.file if self.file
        is not None.
        """
        for filename in filenames:
            if not filename in self.list:
                self.list.append(filename)
                if self.file is not None:
                    self.file.write(filename + '\n')

    def close(self):
        """
        Close the output file.
        """
        self.file.close()
        self.file = None

    def __repr__(self):
        try:
            output_file = self.file.name
        except AttributeError:
            output_file = None
        return '%s(%r, %s)' % (self.__class__.__name__, output_file, self.list)