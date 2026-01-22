import html
import itertools
from contextlib import closing
from inspect import isclass
from io import StringIO
from pathlib import Path
from string import Template
from .. import __version__, config_context
from .fixes import parse_version
@_doc_link_template.setter
def _doc_link_template(self, value):
    setattr(self, '__doc_link_template', value)