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
class Cifti2BrainModel(xml.XmlSerializable):
    """Element representing a mapping of the dimension to vertex or voxels.

    Mapping to vertices of voxels must be specified.

    * Description - Maps a range of indices to surface vertices or voxels when
      IndicesMapToDataType is "CIFTI_INDEX_TYPE_BRAIN_MODELS."
    * Attributes

        * IndexOffset - The matrix index of the first brainordinate of this
          BrainModel. Note that matrix indices are zero-based.
        * IndexCount - Number of surface vertices or voxels in this brain
          model, must be positive.
        * ModelType - Type of model representing the brain structure (surface
          or voxels).  Valid values are listed in the table below.
        * BrainStructure - Identifies the brain structure. Valid values for
          BrainStructure are listed in the table below. However, if the needed
          structure is not listed in the table, a message should be posted to
          the CIFTI Forum so that a standardized name can be created for the
          structure and added to the table.
        * SurfaceNumberOfVertices - When ModelType is CIFTI_MODEL_TYPE_SURFACE
          this attribute contains the actual (or true) number of vertices in
          the surface that is associated with this BrainModel. When this
          BrainModel represents all vertices in the surface, this value is the
          same as IndexCount. When this BrainModel represents only a subset of
          the surface's vertices, IndexCount will be less than this value.

    * Child Elements

        * VertexIndices (0...1)
        * VoxelIndicesIJK (0...1)

    * Text Content: [NA]
    * Parent Element - MatrixIndicesMap

    For ModelType values, see CIFTI_MODEL_TYPES module attribute.

    For BrainStructure values, see CIFTI_BRAIN_STRUCTURES model attribute.

    Attributes
    ----------
    index_offset : int
        Start of the mapping
    index_count : int
        Number of elements in the array to be mapped
    model_type : str
        One of CIFTI_MODEL_TYPES
    brain_structure : str
        One of CIFTI_BRAIN_STRUCTURES
    surface_number_of_vertices : int
        Number of vertices in the surface. Use only for surface-type structure
    voxel_indices_ijk : Cifti2VoxelIndicesIJK, optional
        Indices on the image towards where the array indices are mapped
    vertex_indices : Cifti2VertexIndices, optional
        Indices of the vertices towards where the array indices are mapped
    """

    def __init__(self, index_offset=None, index_count=None, model_type=None, brain_structure=None, n_surface_vertices=None, voxel_indices_ijk=None, vertex_indices=None):
        self.index_offset = index_offset
        self.index_count = index_count
        self.model_type = model_type
        self.brain_structure = brain_structure
        self.surface_number_of_vertices = n_surface_vertices
        self.voxel_indices_ijk = voxel_indices_ijk
        self.vertex_indices = vertex_indices

    @property
    def voxel_indices_ijk(self):
        return self._voxel_indices_ijk

    @voxel_indices_ijk.setter
    def voxel_indices_ijk(self, value):
        self._voxel_indices_ijk = _value_if_klass(value, Cifti2VoxelIndicesIJK)

    @property
    def vertex_indices(self):
        return self._vertex_indices

    @vertex_indices.setter
    def vertex_indices(self, value):
        self._vertex_indices = _value_if_klass(value, Cifti2VertexIndices)

    def _to_xml_element(self):
        brain_model = xml.Element('BrainModel')
        for key in ('IndexOffset', 'IndexCount', 'ModelType', 'BrainStructure', 'SurfaceNumberOfVertices'):
            attr = _underscore(key)
            value = getattr(self, attr)
            if value is not None:
                brain_model.attrib[key] = str(value)
        if self.voxel_indices_ijk:
            brain_model.append(self.voxel_indices_ijk._to_xml_element())
        if self.vertex_indices:
            brain_model.append(self.vertex_indices._to_xml_element())
        return brain_model