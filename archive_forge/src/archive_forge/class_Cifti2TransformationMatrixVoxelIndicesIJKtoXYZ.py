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
class Cifti2TransformationMatrixVoxelIndicesIJKtoXYZ(xml.XmlSerializable):
    """Matrix that translates voxel indices to spatial coordinates

    * Description - Contains a matrix that translates Voxel IJK Indices to
      spatial XYZ coordinates (+X=>right, +Y=>anterior, +Z=> superior). The
      resulting coordinate is the center of the voxel.
    * Attributes

        * MeterExponent - Integer, specifies that the coordinate result from
          the transformation matrix should be multiplied by 10 to this power to
          get the spatial coordinates in meters (e.g., if this is "-3", then
          the transformation matrix is in millimeters).

    * Child Elements: [NA]
    * Text Content - Sixteen floating-point values, in row-major order, that
      form a 4x4 homogeneous transformation matrix.
    * Parent Element - Volume

    Attributes
    ----------
    meter_exponent : int
        See attribute description above.
    matrix : array-like shape (4, 4)
        Affine transformation matrix from voxel indices to RAS space.
    """

    def __init__(self, meter_exponent=None, matrix=None):
        self.meter_exponent = meter_exponent
        self.matrix = matrix

    def _to_xml_element(self):
        if self.matrix is None:
            raise Cifti2HeaderError('TransformationMatrixVoxelIndicesIJKtoXYZ element requires a matrix')
        trans = xml.Element('TransformationMatrixVoxelIndicesIJKtoXYZ')
        trans.attrib['MeterExponent'] = str(self.meter_exponent)
        trans.text = '\n'.join((' '.join(map('{:.10f}'.format, row)) for row in self.matrix))
        return trans