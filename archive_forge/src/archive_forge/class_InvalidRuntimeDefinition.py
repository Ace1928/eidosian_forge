from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import logging
import os
import subprocess
import sys
import threading
from . import comm
import ruamel.yaml as yaml
from six.moves import input
class InvalidRuntimeDefinition(Error):
    """Raised when an inconsistency is found in the runtime definition."""
    pass