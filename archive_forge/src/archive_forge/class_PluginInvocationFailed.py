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
class PluginInvocationFailed(Error):
    """Raised when a plugin invocation returns a non-zero result code."""