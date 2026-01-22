import re
from typing import TYPE_CHECKING, Any, Dict, Generic, List, Optional, Tuple, TypeVar, cast
from docutils import nodes
from docutils.nodes import Node
from docutils.parsers.rst import directives, roles
from sphinx import addnodes
from sphinx.addnodes import desc_signature
from sphinx.util import docutils
from sphinx.util.docfields import DocFieldTransformer, Field, TypedField
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import nested_parse_with_titles
from sphinx.util.typing import OptionSpec
def after_content(self) -> None:
    """
        Called after parsing content. Used to reset information about the
        current directive context on the build environment.
        """
    pass