import os
import pathlib
import random
import tempfile
import pytest
import networkx as nx
from networkx.utils.decorators import (
from networkx.utils.misc import PythonRandomInterface
@open_file('path', 'wb')
def writer_kwarg(self, **kwargs):
    path = kwargs.get('path', None)
    if path is None:
        with tempfile.NamedTemporaryFile('wb+') as fh:
            self.write(fh)
    else:
        self.write(path)