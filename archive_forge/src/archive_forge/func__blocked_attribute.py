import copy
import io
import errno
import os
import re
import subprocess
import sys
import tempfile
import warnings
import pydot
def _blocked_attribute(obj):
    raise AttributeError('A frozendict cannot be modified.')