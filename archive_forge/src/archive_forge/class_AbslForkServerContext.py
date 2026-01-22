import multiprocessing
import os
import platform
import sys
import unittest
from absl import app
from absl import logging
from tensorflow.python.eager import test
class AbslForkServerContext(multiprocessing.context.ForkServerContext):
    _name = 'absl_forkserver'
    Process = AbslForkServerProcess