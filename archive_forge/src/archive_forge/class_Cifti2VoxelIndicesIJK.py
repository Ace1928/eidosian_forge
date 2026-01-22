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
class Cifti2VoxelIndicesIJK(xml.XmlSerializable, MutableSequence):
    """CIFTI-2 VoxelIndicesIJK: Set of voxel indices contained in a structure

    * Description - Identifies the voxels that model a brain structure, or
      participate in a parcel. Note that when this is a child of BrainModel,
      the IndexCount attribute of the BrainModel indicates the number of voxels
      contained in this element.
    * Attributes: [NA]
    * Child Elements: [NA]
    * Text Content - IJK indices (which are zero-based) of each voxel in this
      brain model or parcel, with each index separated by a whitespace
      character. There are three indices per voxel.  If the parent element is
      BrainModel, then the BrainModel element's IndexCount attribute indicates
      the number of triplets (IJK indices) in this element's content.
    * Parent Elements - BrainModel, Parcel

    Each element of this sequence is a triple of integers.
    """

    def __init__(self, indices=None):
        self._indices = []
        if indices is not None:
            self.extend(indices)

    def __len__(self):
        return len(self._indices)

    def __delitem__(self, index):
        if not isinstance(index, int) and len(index) > 1:
            raise NotImplementedError
        del self._indices[index]

    def __getitem__(self, index):
        if isinstance(index, int):
            return self._indices[index]
        elif len(index) == 2:
            if not isinstance(index[0], int):
                raise NotImplementedError
            return self._indices[index[0]][index[1]]
        else:
            raise ValueError('Only row and row,column access is allowed')

    def __setitem__(self, index, value):
        if isinstance(index, int):
            try:
                value = [int(v) for v in value]
                if len(value) != 3:
                    raise ValueError('rows are triples of ints')
                self._indices[index] = value
            except ValueError:
                raise ValueError('value must be a triple of ints')
        elif len(index) == 2:
            try:
                if not isinstance(index[0], int):
                    raise NotImplementedError
                value = int(value)
                self._indices[index[0]][index[1]] = value
            except ValueError:
                raise ValueError('value must be an int')
        else:
            raise ValueError

    def insert(self, index, value):
        if not isinstance(index, int) and len(index) != 1:
            raise ValueError('Only rows can be inserted')
        try:
            value = [int(v) for v in value]
            if len(value) != 3:
                raise ValueError
            self._indices.insert(index, value)
        except ValueError:
            raise ValueError('value must be a triple of int')

    def _to_xml_element(self):
        if len(self) == 0:
            raise Cifti2HeaderError('VoxelIndicesIJK element require an index table')
        vox_ind = xml.Element('VoxelIndicesIJK')
        vox_ind.text = '\n'.join((' '.join([str(v) for v in row]) for row in self._indices))
        return vox_ind