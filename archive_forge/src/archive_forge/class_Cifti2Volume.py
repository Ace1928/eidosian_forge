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
class Cifti2Volume(xml.XmlSerializable):
    """CIFTI-2 volume: information about a volume for mappings that use voxels

    * Description - Provides information about the volume for any mappings that
      use voxels.
    * Attributes

        * VolumeDimensions - Three integer values separated by commas, the
          lengths of the three volume file dimensions that are related to
          spatial coordinates, in number of voxels. Voxel indices (which are
          zero-based) that are used in the mapping that this element applies to
          must be within these dimensions.

    * Child Elements

        * TransformationMatrixVoxelIndicesIJKtoXYZ (1)

    * Text Content: [NA]
    * Parent Element - MatrixIndicesMap

    Attributes
    ----------
    volume_dimensions : array-like shape (3,)
        See attribute description above.
    transformation_matrix_voxel_indices_ijk_to_xyz         : Cifti2TransformationMatrixVoxelIndicesIJKtoXYZ
        Matrix that translates voxel indices to spatial coordinates
    """

    def __init__(self, volume_dimensions=None, transform_matrix=None):
        self.volume_dimensions = volume_dimensions
        self.transformation_matrix_voxel_indices_ijk_to_xyz = transform_matrix

    def _to_xml_element(self):
        if self.volume_dimensions is None:
            raise Cifti2HeaderError('Volume element requires dimensions')
        volume = xml.Element('Volume')
        volume.attrib['VolumeDimensions'] = ','.join([str(val) for val in self.volume_dimensions])
        volume.append(self.transformation_matrix_voxel_indices_ijk_to_xyz._to_xml_element())
        return volume