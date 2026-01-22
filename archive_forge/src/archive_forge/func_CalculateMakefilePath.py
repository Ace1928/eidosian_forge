import os
import re
import subprocess
import gyp
import gyp.common
import gyp.xcode_emulation
from gyp.common import GetEnvironFallback
import hashlib
def CalculateMakefilePath(build_file, base_name):
    """Determine where to write a Makefile for a given gyp file."""
    base_path = gyp.common.RelativePath(os.path.dirname(build_file), options.depth)
    output_file = os.path.join(options.depth, base_path, base_name)
    if options.generator_output:
        output_file = os.path.join(options.depth, options.generator_output, base_path, base_name)
    base_path = gyp.common.RelativePath(os.path.dirname(build_file), options.toplevel_dir)
    return (base_path, output_file)