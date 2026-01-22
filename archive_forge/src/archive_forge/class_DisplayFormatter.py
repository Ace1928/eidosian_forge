import abc
import sys
import traceback
import warnings
from io import StringIO
from decorator import decorator
from traitlets.config.configurable import Configurable
from .getipython import get_ipython
from ..utils.sentinel import Sentinel
from ..utils.dir2 import get_real_method
from ..lib import pretty
from traitlets import (
from typing import Any
class DisplayFormatter(Configurable):
    active_types = List(Unicode(), help='List of currently active mime-types to display.\n        You can use this to set a white-list for formats to display.\n        \n        Most users will not need to change this value.\n        ').tag(config=True)

    @default('active_types')
    def _active_types_default(self):
        return self.format_types

    @observe('active_types')
    def _active_types_changed(self, change):
        for key, formatter in self.formatters.items():
            if key in change['new']:
                formatter.enabled = True
            else:
                formatter.enabled = False
    ipython_display_formatter = ForwardDeclaredInstance('FormatterABC')

    @default('ipython_display_formatter')
    def _default_formatter(self):
        return IPythonDisplayFormatter(parent=self)
    mimebundle_formatter = ForwardDeclaredInstance('FormatterABC')

    @default('mimebundle_formatter')
    def _default_mime_formatter(self):
        return MimeBundleFormatter(parent=self)
    formatters = Dict()

    @default('formatters')
    def _formatters_default(self):
        """Activate the default formatters."""
        formatter_classes = [PlainTextFormatter, HTMLFormatter, MarkdownFormatter, SVGFormatter, PNGFormatter, PDFFormatter, JPEGFormatter, LatexFormatter, JSONFormatter, JavascriptFormatter]
        d = {}
        for cls in formatter_classes:
            f = cls(parent=self)
            d[f.format_type] = f
        return d

    def format(self, obj, include=None, exclude=None):
        """Return a format data dict for an object.

        By default all format types will be computed.

        The following MIME types are usually implemented:

        * text/plain
        * text/html
        * text/markdown
        * text/latex
        * application/json
        * application/javascript
        * application/pdf
        * image/png
        * image/jpeg
        * image/svg+xml

        Parameters
        ----------
        obj : object
            The Python object whose format data will be computed.
        include : list, tuple or set; optional
            A list of format type strings (MIME types) to include in the
            format data dict. If this is set *only* the format types included
            in this list will be computed.
        exclude : list, tuple or set; optional
            A list of format type string (MIME types) to exclude in the format
            data dict. If this is set all format types will be computed,
            except for those included in this argument.
            Mimetypes present in exclude will take precedence over the ones in include

        Returns
        -------
        (format_dict, metadata_dict) : tuple of two dicts
            format_dict is a dictionary of key/value pairs, one of each format that was
            generated for the object. The keys are the format types, which
            will usually be MIME type strings and the values and JSON'able
            data structure containing the raw data for the representation in
            that format.

            metadata_dict is a dictionary of metadata about each mime-type output.
            Its keys will be a strict subset of the keys in format_dict.

        Notes
        -----
            If an object implement `_repr_mimebundle_` as well as various
            `_repr_*_`, the data returned by `_repr_mimebundle_` will take
            precedence and the corresponding `_repr_*_` for this mimetype will
            not be called.

        """
        format_dict = {}
        md_dict = {}
        if self.ipython_display_formatter(obj):
            return ({}, {})
        format_dict, md_dict = self.mimebundle_formatter(obj, include=include, exclude=exclude)
        if format_dict or md_dict:
            if include:
                format_dict = {k: v for k, v in format_dict.items() if k in include}
                md_dict = {k: v for k, v in md_dict.items() if k in include}
            if exclude:
                format_dict = {k: v for k, v in format_dict.items() if k not in exclude}
                md_dict = {k: v for k, v in md_dict.items() if k not in exclude}
        for format_type, formatter in self.formatters.items():
            if format_type in format_dict:
                try:
                    formatter.lookup(obj)
                except KeyError:
                    continue
            if include and format_type not in include:
                continue
            if exclude and format_type in exclude:
                continue
            md = None
            try:
                data = formatter(obj)
            except:
                raise
            if isinstance(data, tuple) and len(data) == 2:
                data, md = data
            if data is not None:
                format_dict[format_type] = data
            if md is not None:
                md_dict[format_type] = md
        return (format_dict, md_dict)

    @property
    def format_types(self):
        """Return the format types (MIME types) of the active formatters."""
        return list(self.formatters.keys())