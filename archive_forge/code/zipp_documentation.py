import io
import posixpath
import zipfile
import itertools
import contextlib
import sys
import pathlib

        Open this entry as text or binary following the semantics
        of ``pathlib.Path.open()`` by passing arguments through
        to io.TextIOWrapper().
        