import subprocess
import sys
from subprocess import PIPE, CalledProcessError
from typing import Any, Dict
import sphinx
from sphinx.application import Sphinx
from sphinx.errors import ExtensionError
from sphinx.locale import __
from sphinx.transforms.post_transforms.images import ImageConverter
from sphinx.util import logging
Converts the image to expected one.