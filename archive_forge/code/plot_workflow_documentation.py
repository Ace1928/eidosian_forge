import sys
import os
import shutil
import io
import re
import textwrap
from os.path import relpath
from errno import EEXIST
import traceback

    Run a nipype workflow creation script and save the graph in *output_dir*.
    Save the images under *output_dir* with file names derived from
    *output_base*
    