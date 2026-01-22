from http://www.nitrc.org/projects/gifti/
from __future__ import annotations
import base64
import sys
import warnings
from copy import copy
from typing import Type, cast
import numpy as np
from .. import xmlutils as xml
from ..caret import CaretMetaData
from ..deprecated import deprecate_with_version
from ..filebasedimages import SerializableImage
from ..nifti1 import data_type_codes, intent_codes, xform_codes
from .util import KIND2FMT, array_index_order_codes, gifti_encoding_codes, gifti_endian_codes
from .parse_gifti_fast import GiftiImageParser
class GiftiCoordSystem(xml.XmlSerializable):
    """Gifti coordinate system transform matrix

    Quotes are from the gifti spec dated 2011-01-14.

        "For a DataArray with an Intent NIFTI_INTENT_POINTSET, this element
        describes the stereotaxic space of the data before and after the
        application of a transformation matrix. The most common stereotaxic
        space is the Talairach Space that places the origin at the anterior
        commissure and the negative X, Y, and Z axes correspond to left,
        posterior, and inferior respectively.  At least one
        CoordinateSystemTransformMatrix is required in a DataArray with an
        intent of NIFTI_INTENT_POINTSET. Multiple
        CoordinateSystemTransformMatrix elements may be used to describe the
        transformation to multiple spaces."

    Attributes
    ----------
    dataspace : int
        From the spec: Contains the stereotaxic space of a DataArray's data
        prior to application of the transformation matrix. The stereotaxic
        space should be one of:

          - NIFTI_XFORM_UNKNOWN
          - NIFTI_XFORM_SCANNER_ANAT
          - NIFTI_XFORM_ALIGNED_ANAT
          - NIFTI_XFORM_TALAIRACH
          - NIFTI_XFORM_MNI_152

    xformspace : int
        Spec: "Contains the stereotaxic space of a DataArray's data after
        application of the transformation matrix. See the DataSpace element for
        a list of stereotaxic spaces."

    xform : array-like shape (4, 4)
        Affine transformation matrix
    """

    def __init__(self, dataspace=0, xformspace=0, xform=None):
        self.dataspace = dataspace
        self.xformspace = xformspace
        if xform is None:
            self.xform = np.identity(4)
        else:
            self.xform = xform

    def __repr__(self):
        src = xform_codes.label[self.dataspace]
        dst = xform_codes.label[self.xformspace]
        return f'<GiftiCoordSystem {src}-to-{dst}>'

    def _to_xml_element(self):
        coord_xform = xml.Element('CoordinateSystemTransformMatrix')
        if self.xform is not None:
            dataspace = xml.SubElement(coord_xform, 'DataSpace')
            dataspace.text = xform_codes.niistring[self.dataspace]
            xformed_space = xml.SubElement(coord_xform, 'TransformedSpace')
            xformed_space.text = xform_codes.niistring[self.xformspace]
            matrix_data = xml.SubElement(coord_xform, 'MatrixData')
            matrix_data.text = _arr2txt(self.xform, '%10.6f')
        return coord_xform

    def print_summary(self):
        print('Dataspace: ', xform_codes.niistring[self.dataspace])
        print('XFormSpace: ', xform_codes.niistring[self.xformspace])
        print('Affine Transformation Matrix: \n', self.xform)