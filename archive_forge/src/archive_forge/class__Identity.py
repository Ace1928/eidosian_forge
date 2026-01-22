import gettext
import os
import re
import textwrap
import warnings
from . import declarative
class _Identity(Validator):

    def __repr__(self):
        return 'validators.Identity'