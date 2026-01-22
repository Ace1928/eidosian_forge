import os
import sys
import zipfile
Splits a path containing a zip file into (zipfile, subpath).
    If there is no zip file, returns (path, None)