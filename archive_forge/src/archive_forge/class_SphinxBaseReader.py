import codecs
import warnings
from typing import TYPE_CHECKING, Any, List, Type
import docutils
from docutils import nodes
from docutils.core import Publisher
from docutils.frontend import Values
from docutils.io import FileInput, Input, NullOutput
from docutils.parsers import Parser
from docutils.parsers.rst import Parser as RSTParser
from docutils.readers import standalone
from docutils.transforms import Transform
from docutils.transforms.references import DanglingReferences
from docutils.writers import UnfilteredWriter
from sphinx import addnodes
from sphinx.deprecation import RemovedInSphinx70Warning
from sphinx.environment import BuildEnvironment
from sphinx.transforms import (AutoIndexUpgrader, DoctreeReadEvent, FigureAligner,
from sphinx.transforms.i18n import (Locale, PreserveTranslatableMessages,
from sphinx.transforms.references import SphinxDomains
from sphinx.util import UnicodeDecodeErrorHandler, get_filetype, logging
from sphinx.util.docutils import LoggingReporter
from sphinx.versioning import UIDTransform
class SphinxBaseReader(standalone.Reader):
    """
    A base class of readers for Sphinx.

    This replaces reporter by Sphinx's on generating document.
    """
    transforms: List[Type[Transform]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        from sphinx.application import Sphinx
        if len(args) > 0 and isinstance(args[0], Sphinx):
            self._app = args[0]
            self._env = self._app.env
            args = args[1:]
        super().__init__(*args, **kwargs)

    def setup(self, app: 'Sphinx') -> None:
        self._app = app
        self._env = app.env

    def get_transforms(self) -> List[Type[Transform]]:
        transforms = super().get_transforms() + self.transforms
        unused = [DanglingReferences]
        for transform in unused:
            if transform in transforms:
                transforms.remove(transform)
        return transforms

    def new_document(self) -> nodes.document:
        """
        Creates a new document object which has a special reporter object good
        for logging.
        """
        document = super().new_document()
        document.__class__ = addnodes.document
        document.transformer = SphinxTransformer(document)
        document.transformer.set_environment(self.settings.env)
        reporter = document.reporter
        document.reporter = LoggingReporter.from_reporter(reporter)
        return document