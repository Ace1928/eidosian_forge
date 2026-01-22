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
class Cifti2NamedMap(xml.XmlSerializable):
    """CIFTI-2 named map: association of name and optional data with a map index

    Associates a name, optional metadata, and possibly a LabelTable with an
    index in a map.

    * Description - Associates a name, optional metadata, and possibly a
      LabelTable with an index in a map.
    * Attributes: [NA]
    * Child Elements

        * MapName (1)
        * LabelTable (0...1)
        * MetaData (0...1)

    * Text Content: [NA]
    * Parent Element - MatrixIndicesMap

    Attributes
    ----------
    map_name : str
        Name of map
    metadata : None or Cifti2MetaData
        Metadata associated with named map
    label_table : None or Cifti2LabelTable
        Label table associated with named map
    """

    def __init__(self, map_name=None, metadata=None, label_table=None):
        self.map_name = map_name
        self.metadata = metadata
        self.label_table = label_table

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, metadata):
        """Set the metadata for this NamedMap

        Parameters
        ----------
        meta : Cifti2MetaData

        Returns
        -------
        None
        """
        self._metadata = _value_if_klass(metadata, Cifti2MetaData)

    @property
    def label_table(self):
        return self._label_table

    @label_table.setter
    def label_table(self, label_table):
        """Set the label_table for this NamedMap

        Parameters
        ----------
        label_table : Cifti2LabelTable

        Returns
        -------
        None
        """
        self._label_table = _value_if_klass(label_table, Cifti2LabelTable)

    def _to_xml_element(self):
        named_map = xml.Element('NamedMap')
        if self.metadata:
            named_map.append(self.metadata._to_xml_element())
        if self.label_table:
            named_map.append(self.label_table._to_xml_element())
        map_name = xml.SubElement(named_map, 'MapName')
        map_name.text = self.map_name
        return named_map