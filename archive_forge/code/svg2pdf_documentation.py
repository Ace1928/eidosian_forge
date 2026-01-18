import base64
import os
import subprocess
import sys
from shutil import which
from tempfile import TemporaryDirectory
from traitlets import List, Unicode, Union, default
from nbconvert.utils.io import FormatSafeDict
from .convertfigures import ConvertFiguresPreprocessor

        Convert a single SVG figure to PDF.  Returns converted data.
        