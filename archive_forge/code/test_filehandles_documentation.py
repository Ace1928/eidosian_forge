import shutil
import unittest
from os.path import join as pjoin
from tempfile import mkdtemp
import numpy as np
from ..loadsave import load, save
from ..nifti1 import Nifti1Image

Check that loading an image does not use up filehandles.
