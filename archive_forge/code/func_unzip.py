from .. import utils
from .._lazyload import requests
import os
import tempfile
import urllib.request
import zipfile
def unzip(filename, destination=None, delete=True):
    """Extract a .zip file and optionally remove the archived version.

    Parameters
    ----------
    filename : string
        Path to the zip file
    destination : string, optional (default: None)
        Path to the folder in which to extract the zip.
        If None, extracts to the same directory the archive is in.
    delete : boolean, optional (default: True)
        If True, deletes the zip file after extraction
    """
    filename = os.path.expanduser(filename)
    if destination is None:
        destination = os.path.dirname(filename)
    elif not os.path.isdir(destination):
        os.mkdir(destination)
    with zipfile.ZipFile(filename, 'r') as handle:
        handle.extractall(destination)
    if delete:
        os.unlink(filename)