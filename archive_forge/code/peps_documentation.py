import sys
import os
import re
import time
from docutils import nodes, utils, languages
from docutils import ApplicationError, DataError
from docutils.transforms import Transform, TransformError
from docutils.transforms import parts, references, misc

        Remove an empty "References" section.

        Called after the `references.TargetNotes` transform is complete.
        