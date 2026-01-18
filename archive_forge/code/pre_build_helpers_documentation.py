import glob
import os
import subprocess
import sys
import tempfile
import textwrap
from setuptools.command.build_ext import customize_compiler, new_compiler
Check basic compilation and linking of C code