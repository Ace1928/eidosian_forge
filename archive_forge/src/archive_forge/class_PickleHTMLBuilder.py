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
class PickleHTMLBuilder(SerializingHTMLBuilder):
    """
    A Builder that dumps the generated HTML into pickle files.
    """
    name = 'pickle'
    epilog = __('You can now process the pickle files in %(outdir)s.')
    implementation = pickle
    implementation_dumps_unicode = False
    additional_dump_args = (pickle.HIGHEST_PROTOCOL,)
    indexer_format = pickle
    indexer_dumps_unicode = False
    out_suffix = '.fpickle'
    globalcontext_filename = 'globalcontext.pickle'
    searchindex_filename = 'searchindex.pickle'