import os
import sys
import hashlib
import filecmp
from collections import namedtuple
from shutil import which
from jedi.cache import memoize_method, time_cache
from jedi.inference.compiled.subprocess import CompiledSubprocess, \
import parso
class InterpreterEnvironment(_SameEnvironmentMixin, _BaseEnvironment):

    def get_inference_state_subprocess(self, inference_state):
        return InferenceStateSameProcess(inference_state)

    def get_sys_path(self):
        return sys.path