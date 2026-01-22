import codecs
import collections
import decimal
import enum
import hashlib
import re
import uuid
from io import BytesIO, FileIO, IOBase
from pathlib import Path
from types import TracebackType
from typing import (
from ._cmap import build_char_map_from_dict
from ._doc_common import PdfDocCommon
from ._encryption import EncryptAlgorithm, Encryption
from ._page import PageObject
from ._page_labels import nums_clear_range, nums_insert, nums_next
from ._reader import PdfReader
from ._utils import (
from .constants import AnnotationDictionaryAttributes as AA
from .constants import CatalogAttributes as CA
from .constants import (
from .constants import CatalogDictionary as CD
from .constants import Core as CO
from .constants import (
from .constants import PageAttributes as PG
from .constants import PagesAttributes as PA
from .constants import TrailerKeys as TK
from .errors import PyPdfError
from .generic import (
from .pagerange import PageRange, PageRangeSpec
from .types import (
from .xmp import XmpInformation
def _insert_filtered_annotations(self, annots: Union[IndirectObject, List[DictionaryObject]], page: PageObject, pages: Dict[int, PageObject], reader: PdfReader) -> List[Destination]:
    outlist = ArrayObject()
    if isinstance(annots, IndirectObject):
        annots = cast('List[Any]', annots.get_object())
    for an in annots:
        ano = cast('DictionaryObject', an.get_object())
        if ano['/Subtype'] != '/Link' or '/A' not in ano or cast('DictionaryObject', ano['/A'])['/S'] != '/GoTo' or ('/Dest' in ano):
            if '/Dest' not in ano:
                outlist.append(self._add_object(ano.clone(self)))
            else:
                d = ano['/Dest']
                if isinstance(d, str):
                    if str(d) in self.get_named_dest_root():
                        outlist.append(ano.clone(self).indirect_reference)
                else:
                    d = cast('ArrayObject', d)
                    p = self._get_cloned_page(d[0], pages, reader)
                    if p is not None:
                        anc = ano.clone(self, ignore_fields=('/Dest',))
                        anc[NameObject('/Dest')] = ArrayObject([p] + d[1:])
                        outlist.append(self._add_object(anc))
        else:
            d = cast('DictionaryObject', ano['/A'])['/D']
            if isinstance(d, str):
                if str(d) in self.get_named_dest_root():
                    outlist.append(ano.clone(self).indirect_reference)
            else:
                d = cast('ArrayObject', d)
                p = self._get_cloned_page(d[0], pages, reader)
                if p is not None:
                    anc = ano.clone(self, ignore_fields=('/D',))
                    cast('DictionaryObject', anc['/A'])[NameObject('/D')] = ArrayObject([p] + d[1:])
                    outlist.append(self._add_object(anc))
    return outlist