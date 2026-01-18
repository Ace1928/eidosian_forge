import os
import sys
from io import BytesIO
from textwrap import dedent
from .. import errors, revision, shelf, shelf_ui, tests
from . import features, script
A variant of Shelver that intercepts console activity, for testing.