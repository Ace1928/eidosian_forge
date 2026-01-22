import base64
import codecs
import datetime
import gzip
import hashlib
from io import BytesIO
import itertools
import logging
import os
import re
import uuid
import numpy as np
from PIL import Image
import matplotlib as mpl
from matplotlib import cbook, font_manager as fm
from matplotlib.backend_bases import (
from matplotlib.backends.backend_mixed import MixedModeRenderer
from matplotlib.colors import rgb2hex
from matplotlib.dates import UTC
from matplotlib.path import Path
from matplotlib import _path
from matplotlib.transforms import Affine2D, Affine2DBase
class FigureCanvasSVG(FigureCanvasBase):
    filetypes = {'svg': 'Scalable Vector Graphics', 'svgz': 'Scalable Vector Graphics'}
    fixed_dpi = 72

    def print_svg(self, filename, *, bbox_inches_restore=None, metadata=None):
        """
        Parameters
        ----------
        filename : str or path-like or file-like
            Output target; if a string, a file will be opened for writing.

        metadata : dict[str, Any], optional
            Metadata in the SVG file defined as key-value pairs of strings,
            datetimes, or lists of strings, e.g., ``{'Creator': 'My software',
            'Contributor': ['Me', 'My Friend'], 'Title': 'Awesome'}``.

            The standard keys and their value types are:

            * *str*: ``'Coverage'``, ``'Description'``, ``'Format'``,
              ``'Identifier'``, ``'Language'``, ``'Relation'``, ``'Source'``,
              ``'Title'``, and ``'Type'``.
            * *str* or *list of str*: ``'Contributor'``, ``'Creator'``,
              ``'Keywords'``, ``'Publisher'``, and ``'Rights'``.
            * *str*, *date*, *datetime*, or *tuple* of same: ``'Date'``. If a
              non-*str*, then it will be formatted as ISO 8601.

            Values have been predefined for ``'Creator'``, ``'Date'``,
            ``'Format'``, and ``'Type'``. They can be removed by setting them
            to `None`.

            Information is encoded as `Dublin Core Metadata`__.

            .. _DC: https://www.dublincore.org/specifications/dublin-core/

            __ DC_
        """
        with cbook.open_file_cm(filename, 'w', encoding='utf-8') as fh:
            if not cbook.file_requires_unicode(fh):
                fh = codecs.getwriter('utf-8')(fh)
            dpi = self.figure.dpi
            self.figure.dpi = 72
            width, height = self.figure.get_size_inches()
            w, h = (width * 72, height * 72)
            renderer = MixedModeRenderer(self.figure, width, height, dpi, RendererSVG(w, h, fh, image_dpi=dpi, metadata=metadata), bbox_inches_restore=bbox_inches_restore)
            self.figure.draw(renderer)
            renderer.finalize()

    def print_svgz(self, filename, **kwargs):
        with cbook.open_file_cm(filename, 'wb') as fh, gzip.GzipFile(mode='w', fileobj=fh) as gzipwriter:
            return self.print_svg(gzipwriter, **kwargs)

    def get_default_filetype(self):
        return 'svg'

    def draw(self):
        self.figure.draw_without_rendering()
        return super().draw()