import os
import importlib.util
import sys
from distutils.core import Command
from distutils.errors import DistutilsOptionError
def _mutate_outputs(self, has_any, build_cmd, cmd_option, output_dir):
    if not has_any:
        return []
    build_cmd = self.get_finalized_command(build_cmd)
    build_files = build_cmd.get_outputs()
    build_dir = getattr(build_cmd, cmd_option)
    prefix_len = len(build_dir) + len(os.sep)
    outputs = []
    for file in build_files:
        outputs.append(os.path.join(output_dir, file[prefix_len:]))
    return outputs