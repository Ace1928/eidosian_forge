import abc
import collections
import contextlib
import functools
import importlib
import subprocess
import typing
import warnings
from typing import Union, Iterable, Dict, Optional, Callable, Type
from qiskit.exceptions import MissingOptionalLibraryError, OptionalDependencyImportWarning
from .classtools import wrap_method

        Args:
            command: the strings that make up the command to be run.  For example,
                ``["pdflatex", "-version"]``.

        Raises:
            ValueError: if an empty command is given.
        