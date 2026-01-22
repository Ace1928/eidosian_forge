import re
import sys
import os.path as op
from glob import glob
from traits.trait_errors import TraitError
from nipype.utils.filemanip import loadcrash
display crash file content and rerun if required