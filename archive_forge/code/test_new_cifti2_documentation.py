import numpy as np
import pytest
import nibabel as nib
from nibabel import cifti2 as ci
from nibabel.tmpdirs import InTemporaryDirectory
from ...testing import (
Tests the generation of new CIFTI-2 files from scratch

Contains a series of functions to create and check each of the 5 CIFTI index
types (i.e. BRAIN_MODELS, PARCELS, SCALARS, LABELS, and SERIES).

These functions are used in the tests to generate most CIFTI file types from
scratch.
