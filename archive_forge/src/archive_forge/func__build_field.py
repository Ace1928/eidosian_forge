import struct
import zlib
from abc import abstractmethod
from datetime import datetime
from typing import (
from ._encryption import Encryption
from ._page import PageObject, _VirtualList
from ._page_labels import index2label as page_index2page_label
from ._utils import (
from .constants import CatalogAttributes as CA
from .constants import CatalogDictionary as CD
from .constants import (
from .constants import Core as CO
from .constants import DocumentInformationAttributes as DI
from .constants import FieldDictionaryAttributes as FA
from .constants import PageAttributes as PG
from .constants import PagesAttributes as PA
from .errors import (
from .generic import (
from .types import OutlineType, PagemodeType
from .xmp import XmpInformation
def _build_field(self, field: Union[TreeObject, DictionaryObject], retval: Dict[Any, Any], fileobj: Any, field_attributes: Any) -> None:
    self._check_kids(field, retval, fileobj)
    try:
        key = cast(str, field['/TM'])
    except KeyError:
        try:
            if '/Parent' in field:
                key = self._get_qualified_field_name(cast(DictionaryObject, field['/Parent'])) + '.'
            else:
                key = ''
            key += cast(str, field['/T'])
        except KeyError:
            return
    if fileobj:
        self._write_field(fileobj, field, field_attributes)
        fileobj.write('\n')
    retval[key] = Field(field)
    obj = retval[key].indirect_reference.get_object()
    if obj.get(FA.FT, '') == '/Ch':
        retval[key][NameObject('/_States_')] = obj[NameObject(FA.Opt)]
    if obj.get(FA.FT, '') == '/Btn' and '/AP' in obj:
        retval[key][NameObject('/_States_')] = ArrayObject(list(obj['/AP']['/N'].keys()))
        if '/Off' not in retval[key]['/_States_']:
            retval[key][NameObject('/_States_')].append(NameObject('/Off'))
    elif obj.get(FA.FT, '') == '/Btn' and obj.get(FA.Ff, 0) & FA.FfBits.Radio != 0:
        states: List[str] = []
        retval[key][NameObject('/_States_')] = ArrayObject(states)
        for k in obj.get(FA.Kids, {}):
            k = k.get_object()
            for s in list(k['/AP']['/N'].keys()):
                if s not in states:
                    states.append(s)
            retval[key][NameObject('/_States_')] = ArrayObject(states)
        if obj.get(FA.Ff, 0) & FA.FfBits.NoToggleToOff != 0 and '/Off' in retval[key]['/_States_']:
            del retval[key]['/_States_'][retval[key]['/_States_'].index('/Off')]