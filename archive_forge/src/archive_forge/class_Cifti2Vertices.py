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
class Cifti2Vertices(xml.XmlSerializable, MutableSequence):
    """CIFTI-2 vertices - association of brain structure and a list of vertices

    * Description - Contains a BrainStructure type and a list of vertex indices
      within a Parcel.
    * Attributes

        * BrainStructure - A string from the BrainStructure list to identify
          what surface this vertex list is from (usually left cortex, right
          cortex, or cerebellum).

    * Child Elements: [NA]
    * Text Content - Vertex indices (which are independent for each surface,
      and zero-based) separated by whitespace characters.
    * Parent Element - Parcel

    The class behaves like a list of Vertex indices (which are independent for
    each surface, and zero-based)

    Attributes
    ----------
    brain_structure : str
        A string from the BrainStructure list to identify what surface this
        vertex list is from (usually left cortex, right cortex, or cerebellum).
    """

    def __init__(self, brain_structure=None, vertices=None):
        self._vertices = []
        if vertices is not None:
            self.extend(vertices)
        self.brain_structure = brain_structure

    def __len__(self):
        return len(self._vertices)

    def __delitem__(self, index):
        del self._vertices[index]

    def __getitem__(self, index):
        return self._vertices[index]

    def __setitem__(self, index, value):
        try:
            value = int(value)
            self._vertices[index] = value
        except ValueError:
            raise ValueError('value must be an int')

    def insert(self, index, value):
        try:
            value = int(value)
            self._vertices.insert(index, value)
        except ValueError:
            raise ValueError('value must be an int')

    def _to_xml_element(self):
        if self.brain_structure is None:
            raise Cifti2HeaderError('Vertices element require a BrainStructure')
        vertices = xml.Element('Vertices')
        vertices.attrib['BrainStructure'] = str(self.brain_structure)
        vertices.text = ' '.join([str(i) for i in self])
        return vertices