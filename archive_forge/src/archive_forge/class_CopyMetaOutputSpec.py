import os
from os import path as op
import string
import errno
from glob import glob
import nibabel as nb
import imghdr
from .base import (
class CopyMetaOutputSpec(TraitedSpec):
    dest_file = File(exists=True)