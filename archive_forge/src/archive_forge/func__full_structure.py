import re
from collections import OrderedDict
from collections.abc import Iterable, MutableMapping, MutableSequence
from warnings import warn
import numpy as np
from .. import xmlutils as xml
from ..arrayproxy import reshape_dataobj
from ..caret import CaretMetaData
from ..dataobj_images import DataobjImage
from ..filebasedimages import FileBasedHeader, SerializableImage
from ..nifti1 import Nifti1Extensions
from ..nifti2 import Nifti2Header, Nifti2Image
from ..volumeutils import Recoder, make_dt_codes
def _full_structure(struct: str):
    """Expands STRUCT_NAME into:

    STRUCT_NAME, CIFTI_STRUCTURE_STRUCT_NAME, StructName
    """
    return (struct, f'CIFTI_STRUCTURE_{struct}', ''.join((word.capitalize() for word in struct.split('_'))))