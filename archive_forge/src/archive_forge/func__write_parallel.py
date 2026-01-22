import codecs
import pickle
import time
import warnings
from os import path
from typing import (TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple,
from docutils import nodes
from docutils.nodes import Node
from docutils.utils import DependencyList
from sphinx.config import Config
from sphinx.deprecation import RemovedInSphinx70Warning
from sphinx.environment import CONFIG_CHANGED_REASON, CONFIG_OK, BuildEnvironment
from sphinx.environment.adapters.asset import ImageAdapter
from sphinx.errors import SphinxError
from sphinx.events import EventManager
from sphinx.locale import __
from sphinx.util import (UnicodeDecodeErrorHandler, get_filetype, import_object, logging,
from sphinx.util.build_phase import BuildPhase
from sphinx.util.console import bold  # type: ignore
from sphinx.util.docutils import sphinx_domains
from sphinx.util.i18n import CatalogInfo, CatalogRepository, docname_to_domain
from sphinx.util.osutil import SEP, ensuredir, relative_uri, relpath
from sphinx.util.parallel import ParallelTasks, SerialTasks, make_chunks, parallel_available
from sphinx.util.tags import Tags
from sphinx.util.typing import NoneType
from sphinx import directives  # NOQA isort:skip
from sphinx import roles  # NOQA isort:skip
def _write_parallel(self, docnames: Sequence[str], nproc: int) -> None:

    def write_process(docs: List[Tuple[str, nodes.document]]) -> None:
        self.app.phase = BuildPhase.WRITING
        for docname, doctree in docs:
            self.write_doc(docname, doctree)
    firstname, docnames = (docnames[0], docnames[1:])
    self.app.phase = BuildPhase.RESOLVING
    doctree = self.env.get_and_resolve_doctree(firstname, self)
    self.app.phase = BuildPhase.WRITING
    self.write_doc_serialized(firstname, doctree)
    self.write_doc(firstname, doctree)
    tasks = ParallelTasks(nproc)
    chunks = make_chunks(docnames, nproc)
    progress = status_iterator(chunks, __('writing output... '), 'darkgreen', len(chunks), self.app.verbosity)

    def on_chunk_done(args: List[Tuple[str, NoneType]], result: NoneType) -> None:
        next(progress)
    self.app.phase = BuildPhase.RESOLVING
    for chunk in chunks:
        arg = []
        for docname in chunk:
            doctree = self.env.get_and_resolve_doctree(docname, self)
            self.write_doc_serialized(docname, doctree)
            arg.append((docname, doctree))
        tasks.add_task(write_process, arg, on_chunk_done)
    tasks.join()
    logger.info('')