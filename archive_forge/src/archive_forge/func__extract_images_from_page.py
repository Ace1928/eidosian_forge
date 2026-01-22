from __future__ import annotations
import warnings
from typing import (
from urllib.parse import urlparse
import numpy as np
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob
def _extract_images_from_page(self, page: pdfplumber.page.Page) -> str:
    """Extract images from page and get the text with RapidOCR."""
    if not self.extract_images:
        return ''
    images = []
    for img in page.images:
        if img['stream']['Filter'].name in _PDF_FILTER_WITHOUT_LOSS:
            images.append(np.frombuffer(img['stream'].get_data(), dtype=np.uint8).reshape(img['stream']['Height'], img['stream']['Width'], -1))
        elif img['stream']['Filter'].name in _PDF_FILTER_WITH_LOSS:
            images.append(img['stream'].get_data())
        else:
            warnings.warn('Unknown PDF Filter!')
    return extract_from_images_with_rapidocr(images)