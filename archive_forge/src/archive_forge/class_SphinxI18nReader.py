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
class SphinxI18nReader(SphinxBaseReader):
    """
    A document reader for i18n.

    This returns the source line number of original text as current source line number
    to let users know where the error happened.
    Because the translated texts are partial and they don't have correct line numbers.
    """

    def setup(self, app: 'Sphinx') -> None:
        super().setup(app)
        self.transforms = self.transforms + app.registry.get_transforms()
        unused = [PreserveTranslatableMessages, Locale, RemoveTranslatableInline, AutoIndexUpgrader, FigureAligner, SphinxDomains, DoctreeReadEvent, UIDTransform]
        for transform in unused:
            if transform in self.transforms:
                self.transforms.remove(transform)