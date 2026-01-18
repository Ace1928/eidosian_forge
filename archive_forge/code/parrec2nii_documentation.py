import csv
import os
import sys
from optparse import Option, OptionParser
import numpy as np
import numpy.linalg as npl
import nibabel
import nibabel.nifti1 as nifti1
import nibabel.parrec as pr
from nibabel.affines import apply_affine, from_matvec, to_matvec
from nibabel.filename_parser import splitext_addext
from nibabel.mriutils import MRIError, calculate_dwell_time
from nibabel.orientations import apply_orientation, inv_ornt_aff, io_orientation
from nibabel.parrec import one_line
from nibabel.volumeutils import fname_ext_ul_case
Code for PAR/REC to NIfTI converter command
