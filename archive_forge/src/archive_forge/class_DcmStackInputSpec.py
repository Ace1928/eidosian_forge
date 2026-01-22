import os
from os import path as op
import string
import errno
from glob import glob
import nibabel as nb
import imghdr
from .base import (
class DcmStackInputSpec(NiftiGeneratorBaseInputSpec):
    dicom_files = traits.Either(InputMultiPath(File(exists=True)), Directory(exists=True), traits.Str(), mandatory=True)
    embed_meta = traits.Bool(desc='Embed DICOM meta data into result')
    exclude_regexes = traits.List(desc='Meta data to exclude, suplementing any default exclude filters')
    include_regexes = traits.List(desc='Meta data to include, overriding any exclude filters')
    force_read = traits.Bool(True, usedefault=True, desc='Force reading files without DICM marker')