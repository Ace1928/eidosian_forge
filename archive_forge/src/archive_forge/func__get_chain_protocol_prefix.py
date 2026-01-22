import dataclasses
import glob as py_glob
import io
import os
import os.path
import sys
import tempfile
from tensorboard.compat.tensorflow_stub import compat, errors
def _get_chain_protocol_prefix(self, filename):
    chain_prefix, chain_sep, last_path = filename.rpartition(self.CHAIN_SEPARATOR)
    protocol, sep, _ = last_path.rpartition(self.SEPARATOR)
    return chain_prefix + chain_sep + protocol + sep