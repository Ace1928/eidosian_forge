import os
import posixpath
import sys
import warnings
from pyarrow.util import doc, _DEPR_MSG
from pyarrow.filesystem import FileSystem
import pyarrow._hdfsio as _hdfsio

        Directory tree generator for HDFS, like os.walk.

        Parameters
        ----------
        top_path : str
            Root directory for tree traversal.

        Returns
        -------
        Generator yielding 3-tuple (dirpath, dirnames, filename)
        