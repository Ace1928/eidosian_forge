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
class Cifti2Surface(xml.XmlSerializable):
    """Cifti surface: association of brain structure and number of vertices

    * Description - Specifies the number of vertices for a surface, when
      IndicesMapToDataType is "CIFTI_INDEX_TYPE_PARCELS." This is separate from
      the Parcel element because there can be multiple parcels on one surface,
      and one parcel may involve multiple surfaces.
    * Attributes

        * BrainStructure - A string from the BrainStructure list to identify
          what surface structure this element refers to (usually left cortex,
          right cortex, or cerebellum).
        * SurfaceNumberOfVertices - The number of vertices that this
          structure's surface contains.

    * Child Elements: [NA]
    * Text Content: [NA]
    * Parent Element - MatrixIndicesMap

    Attributes
    ----------
    brain_structure : str
        Name of brain structure
    surface_number_of_vertices : int
        Number of vertices on surface
    """

    def __init__(self, brain_structure=None, surface_number_of_vertices=None):
        self.brain_structure = brain_structure
        self.surface_number_of_vertices = surface_number_of_vertices

    def _to_xml_element(self):
        if self.brain_structure is None:
            raise Cifti2HeaderError('Surface element requires at least 1 BrainStructure')
        surf = xml.Element('Surface')
        surf.attrib['BrainStructure'] = str(self.brain_structure)
        surf.attrib['SurfaceNumberOfVertices'] = str(self.surface_number_of_vertices)
        return surf