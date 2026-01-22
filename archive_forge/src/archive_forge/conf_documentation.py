import os
import sys
import sys
from os.path import basename
import sphinx
from docutils import nodes, statemachine
from docutils.parsers.rst import Directive
Execute the specified python code and insert the output into the document