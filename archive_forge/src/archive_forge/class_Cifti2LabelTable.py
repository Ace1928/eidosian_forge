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
class Cifti2LabelTable(xml.XmlSerializable, MutableMapping):
    """CIFTI-2 label table: a sequence of ``Cifti2Label``\\s

    * Description - Used by NamedMap when IndicesMapToDataType is
      "CIFTI_INDEX_TYPE_LABELS" in order to associate names and display colors
      with label keys. Note that LABELS is the only mapping type that uses a
      LabelTable. Display coloring of continuous-valued data is not specified
      by CIFTI-2.
    * Attributes: [NA]
    * Child Elements

        * Label (0...N)

    * Text Content: [NA]
    * Parent Element - NamedMap
    """

    def __init__(self):
        self._labels = OrderedDict()

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, key):
        return self._labels[key]

    def append(self, label):
        self[label.key] = label

    def __setitem__(self, key, value):
        if isinstance(value, Cifti2Label):
            if key != value.key:
                raise ValueError("The key and the label's key must agree")
            self._labels[key] = value
            return
        if len(value) != 5:
            raise ValueError('Value should be length 5')
        try:
            self._labels[key] = Cifti2Label(*[key] + list(value))
        except ValueError:
            raise ValueError('Key should be int, value should be sequence of str and 4 floats between 0 and 1')

    def __delitem__(self, key):
        del self._labels[key]

    def __iter__(self):
        return iter(self._labels)

    def _to_xml_element(self):
        if len(self) == 0:
            raise Cifti2HeaderError('LabelTable element requires at least 1 label')
        labeltable = xml.Element('LabelTable')
        for ele in self._labels.values():
            labeltable.append(ele._to_xml_element())
        return labeltable