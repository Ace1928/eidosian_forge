import sys
import os
import os.path
import tempfile
import zipfile
from xml.dom import minidom
import time
import re
import copy
import itertools
import docutils
from docutils import frontend, nodes, utils, writers, languages
from docutils.readers import standalone
from docutils.transforms import references
def ToString(et):
    outstream = StringIO()
    if sys.version_info >= (3, 2):
        et.write(outstream, encoding='unicode')
    else:
        et.write(outstream)
    s1 = outstream.getvalue()
    outstream.close()
    return s1