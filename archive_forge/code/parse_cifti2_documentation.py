from io import BytesIO
import numpy as np
from packaging.version import Version, parse
from .. import xmlutils as xml
from ..batteryrunners import Report
from ..nifti1 import Nifti1Extension, extension_codes, intent_codes
from ..nifti2 import Nifti2Header, Nifti2Image
from ..spatialimages import HeaderDataError
from .cifti2 import (
True if there is character data pending for processing