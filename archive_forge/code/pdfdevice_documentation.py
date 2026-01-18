from typing import (
from pdfminer.psparser import PSLiteral
from . import utils
from .pdfcolor import PDFColorSpace
from .pdffont import PDFFont
from .pdffont import PDFUnicodeNotDefined
from .pdfpage import PDFPage
from .pdftypes import PDFStream
from .utils import Matrix, Point, Rect, PathSegment
Translate the output of PDFPageInterpreter to the output that is needed