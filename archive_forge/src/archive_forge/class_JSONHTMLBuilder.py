from __future__ import annotations
import os
import pickle
import types
from os import path
from typing import Any
from sphinx.application import ENV_PICKLE_FILENAME, Sphinx
from sphinx.builders.html import BuildInfo, StandaloneHTMLBuilder
from sphinx.locale import get_translation
from sphinx.util.osutil import SEP, copyfile, ensuredir, os_path
from sphinxcontrib.serializinghtml import jsonimpl
class JSONHTMLBuilder(SerializingHTMLBuilder):
    """
    A builder that dumps the generated HTML into JSON files.
    """
    name = 'json'
    epilog = __('You can now process the JSON files in %(outdir)s.')
    implementation = jsonimpl
    implementation_dumps_unicode = True
    indexer_format = jsonimpl
    indexer_dumps_unicode = True
    out_suffix = '.fjson'
    globalcontext_filename = 'globalcontext.json'
    searchindex_filename = 'searchindex.json'