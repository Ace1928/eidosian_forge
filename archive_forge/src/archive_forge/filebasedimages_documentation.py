from __future__ import annotations
import io
import typing as ty
from copy import deepcopy
from urllib import request
from ._compression import COMPRESSION_ERRORS
from .fileholders import FileHolder, FileMap
from .filename_parser import TypesFilenamesError, _stringify_path, splitext_addext, types_filenames
from .openers import ImageOpener
Retrieve and load an image from a URL

        Class method

        Parameters
        ----------
        url : str or urllib.request.Request object
            URL of file to retrieve
        timeout : float, optional
            Time (in seconds) to wait for a response
        