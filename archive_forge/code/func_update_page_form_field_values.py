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
def update_page_form_field_values(self, page: Union[PageObject, List[PageObject], None], fields: Dict[str, Any], flags: FieldFlag=OPTIONAL_READ_WRITE_FIELD, auto_regenerate: Optional[bool]=True) -> None:
    """
        Update the form field values for a given page from a fields dictionary.

        Copy field texts and values from fields to page.
        If the field links to a parent object, add the information to the parent.

        Args:
            page: `PageObject` - references **PDF writer's page** where the
                annotations and field data will be updated.
                `List[Pageobject]` - provides list of page to be processsed.
                `None` - all pages.
            fields: a Python dictionary of field names (/T) and text
                values (/V).
            flags: An integer (0 to 7). The first bit sets ReadOnly, the
                second bit sets Required, the third bit sets NoExport. See
                PDF Reference Table 8.70 for details.
            auto_regenerate: set/unset the need_appearances flag ;
                the flag is unchanged if auto_regenerate is None.
        """
    if CatalogDictionary.ACRO_FORM not in self._root_object:
        raise PyPdfError('No /AcroForm dictionary in PdfWriter Object')
    af = cast(DictionaryObject, self._root_object[CatalogDictionary.ACRO_FORM])
    if InteractiveFormDictEntries.Fields not in af:
        raise PyPdfError('No /Fields dictionary in Pdf in PdfWriter Object')
    if isinstance(auto_regenerate, bool):
        self.set_need_appearances_writer(auto_regenerate)
    if page is None:
        page = list(self.pages)
    if isinstance(page, list):
        for p in page:
            if PG.ANNOTS in p:
                self.update_page_form_field_values(p, fields, flags, None)
        return None
    if PG.ANNOTS not in page:
        logger_warning('No fields to update on this page', __name__)
        return
    for writer_annot in page[PG.ANNOTS]:
        writer_annot = cast(DictionaryObject, writer_annot.get_object())
        if writer_annot.get('/Subtype', '') != '/Widget':
            continue
        if '/FT' in writer_annot and '/T' in writer_annot:
            writer_parent_annot = writer_annot
        else:
            writer_parent_annot = writer_annot.get(PG.PARENT, DictionaryObject()).get_object()
        for field, value in fields.items():
            if not (self._get_qualified_field_name(writer_parent_annot) == field or writer_parent_annot.get('/T', None) == field):
                continue
            if flags:
                writer_annot[NameObject(FA.Ff)] = NumberObject(flags)
            if isinstance(value, list):
                lst = ArrayObject((TextStringObject(v) for v in value))
                writer_parent_annot[NameObject(FA.V)] = lst
            else:
                writer_parent_annot[NameObject(FA.V)] = TextStringObject(value)
            if writer_parent_annot.get(FA.FT) in '/Btn':
                v = NameObject(value)
                if v not in writer_annot[NameObject(AA.AP)][NameObject('/N')]:
                    v = NameObject('/Off')
                writer_annot[NameObject(AA.AS)] = v
            elif writer_parent_annot.get(FA.FT) == '/Tx' or writer_parent_annot.get(FA.FT) == '/Ch':
                self._update_field_annotation(writer_parent_annot, writer_annot)
            elif writer_annot.get(FA.FT) == '/Sig':
                logger_warning('Signature forms not implemented yet', __name__)