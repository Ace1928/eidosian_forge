from __future__ import annotations
import warnings
from typing import (
from urllib.parse import urlparse
import numpy as np
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob
class PyPDFium2Parser(BaseBlobParser):
    """Parse `PDF` with `PyPDFium2`."""

    def __init__(self, extract_images: bool=False) -> None:
        """Initialize the parser."""
        try:
            import pypdfium2
        except ImportError:
            raise ImportError('pypdfium2 package not found, please install it with `pip install pypdfium2`')
        self.extract_images = extract_images

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        import pypdfium2
        with blob.as_bytes_io() as file_path:
            pdf_reader = pypdfium2.PdfDocument(file_path, autoclose=True)
            try:
                for page_number, page in enumerate(pdf_reader):
                    text_page = page.get_textpage()
                    content = text_page.get_text_range()
                    text_page.close()
                    content += '\n' + self._extract_images_from_page(page)
                    page.close()
                    metadata = {'source': blob.source, 'page': page_number}
                    yield Document(page_content=content, metadata=metadata)
            finally:
                pdf_reader.close()

    def _extract_images_from_page(self, page: pypdfium2._helpers.page.PdfPage) -> str:
        """Extract images from page and get the text with RapidOCR."""
        if not self.extract_images:
            return ''
        import pypdfium2.raw as pdfium_c
        images = list(page.get_objects(filter=(pdfium_c.FPDF_PAGEOBJ_IMAGE,)))
        images = list(map(lambda x: x.get_bitmap().to_numpy(), images))
        return extract_from_images_with_rapidocr(images)