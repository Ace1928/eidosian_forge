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
class Cifti2Parcel(xml.XmlSerializable):
    """CIFTI-2 parcel: association of a name with vertices and/or voxels

    * Description - Associates a name, plus vertices and/or voxels, with an
      index.
    * Attributes

        * Name - The name of the parcel

    * Child Elements

        * Vertices (0...N)
        * VoxelIndicesIJK (0...1)

    * Text Content: [NA]
    * Parent Element - MatrixIndicesMap

    Attributes
    ----------
    name : str
        Name of parcel
    voxel_indices_ijk : None or Cifti2VoxelIndicesIJK
        Voxel indices associated with parcel
    vertices : list of Cifti2Vertices
        Vertices associated with parcel
    """

    def __init__(self, name=None, voxel_indices_ijk=None, vertices=None):
        self.name = name
        self._voxel_indices_ijk = voxel_indices_ijk
        self.vertices = vertices if vertices is not None else []
        for val in self.vertices:
            if not isinstance(val, Cifti2Vertices):
                raise ValueError('Cifti2Parcel vertices must be instances of Cifti2Vertices')

    @property
    def voxel_indices_ijk(self):
        return self._voxel_indices_ijk

    @voxel_indices_ijk.setter
    def voxel_indices_ijk(self, value):
        self._voxel_indices_ijk = _value_if_klass(value, Cifti2VoxelIndicesIJK)

    def append_cifti_vertices(self, vertices):
        """Appends a Cifti2Vertices element to the Cifti2Parcel

        Parameters
        ----------
        vertices : Cifti2Vertices
        """
        if not isinstance(vertices, Cifti2Vertices):
            raise TypeError('Not a valid Cifti2Vertices instance')
        self.vertices.append(vertices)

    def pop_cifti2_vertices(self, ith):
        """Pops the ith vertices element from the Cifti2Parcel"""
        self.vertices.pop(ith)

    def _to_xml_element(self):
        if self.name is None:
            raise Cifti2HeaderError('Parcel element requires a name')
        parcel = xml.Element('Parcel')
        parcel.attrib['Name'] = str(self.name)
        if self.voxel_indices_ijk:
            parcel.append(self.voxel_indices_ijk._to_xml_element())
        for vertex in self.vertices:
            parcel.append(vertex._to_xml_element())
        return parcel