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
def _add_articles_thread(self, thread: DictionaryObject, pages: Dict[int, PageObject], reader: PdfReader) -> IndirectObject:
    """
        Clone the thread with only the applicable articles.

        Args:
            thread:
            pages:
            reader:

        Returns:
            The added thread as an indirect reference
        """
    nthread = thread.clone(self, force_duplicate=True, ignore_fields=('/F',))
    self.threads.append(nthread.indirect_reference)
    first_article = cast('DictionaryObject', thread['/F'])
    current_article: Optional[DictionaryObject] = first_article
    new_article: Optional[DictionaryObject] = None
    while current_article is not None:
        pag = self._get_cloned_page(cast('PageObject', current_article['/P']), pages, reader)
        if pag is not None:
            if new_article is None:
                new_article = cast('DictionaryObject', self._add_object(DictionaryObject()).get_object())
                new_first = new_article
                nthread[NameObject('/F')] = new_article.indirect_reference
            else:
                new_article2 = cast('DictionaryObject', self._add_object(DictionaryObject({NameObject('/V'): new_article.indirect_reference})).get_object())
                new_article[NameObject('/N')] = new_article2.indirect_reference
                new_article = new_article2
            new_article[NameObject('/P')] = pag
            new_article[NameObject('/T')] = nthread.indirect_reference
            new_article[NameObject('/R')] = current_article['/R']
            pag_obj = cast('PageObject', pag.get_object())
            if '/B' not in pag_obj:
                pag_obj[NameObject('/B')] = ArrayObject()
            cast('ArrayObject', pag_obj['/B']).append(new_article.indirect_reference)
        current_article = cast('DictionaryObject', current_article['/N'])
        if current_article == first_article:
            new_article[NameObject('/N')] = new_first.indirect_reference
            new_first[NameObject('/V')] = new_article.indirect_reference
            current_article = None
    assert nthread.indirect_reference is not None
    return nthread.indirect_reference