import json
import re
import time
from mako import ast
from mako import exceptions
from mako import filters
from mako import parsetree
from mako import util
from mako.pygen import PythonPrinter
update the state of this Identifiers with the undeclared
        and declared identifiers of the given node.