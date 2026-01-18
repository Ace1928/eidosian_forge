import os
import pathlib
import random
import tempfile
import pytest
import networkx as nx
from networkx.utils.decorators import (
from networkx.utils.misc import PythonRandomInterface
@staticmethod
@open_file(0, 'wb')
def writer_arg0(path):
    path.write(b'demo')