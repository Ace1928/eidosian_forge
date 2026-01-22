import ast
import gyp.common
import gyp.simple_copy
import multiprocessing
import os.path
import re
import shlex
import signal
import subprocess
import sys
import threading
import traceback
from distutils.version import StrictVersion
from gyp.common import GypError
from gyp.common import OrderedSet
class ParallelState:
    """Class to keep track of state when processing input files in parallel.

  If build files are loaded in parallel, use this to keep track of
  state during farming out and processing parallel jobs. It's stored
  in a global so that the callback function can have access to it.
  """

    def __init__(self):
        self.pool = None
        self.condition = None
        self.data = None
        self.pending = 0
        self.scheduled = set()
        self.dependencies = []
        self.error = False

    def LoadTargetBuildFileCallback(self, result):
        """Handle the results of running LoadTargetBuildFile in another process.
    """
        self.condition.acquire()
        if not result:
            self.error = True
            self.condition.notify()
            self.condition.release()
            return
        build_file_path0, build_file_data0, dependencies0 = result
        self.data[build_file_path0] = build_file_data0
        self.data['target_build_files'].add(build_file_path0)
        for new_dependency in dependencies0:
            if new_dependency not in self.scheduled:
                self.scheduled.add(new_dependency)
                self.dependencies.append(new_dependency)
        self.pending -= 1
        self.condition.notify()
        self.condition.release()