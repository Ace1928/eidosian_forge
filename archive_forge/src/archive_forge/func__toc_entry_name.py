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
def _toc_entry_name(self, sig_node: desc_signature) -> str:
    """
        Returns the text of the table of contents entry for the object.

        This function is called once, in :py:meth:`run`, to set the name for the
        table of contents entry (a special attribute ``_toc_name`` is set on the
        object node, later used in
        ``environment.collectors.toctree.TocTreeCollector.process_doc().build_toc()``
        when the table of contents entries are collected).

        To support table of contents entries for their objects, domains must
        override this method, also respecting the configuration setting
        ``toc_object_entries_show_parents``. Domains must also override
        :py:meth:`_object_hierarchy_parts`, with one (string) entry for each part of the
        object's hierarchy. The result of this method is set on the signature
        node, and can be accessed as ``sig_node['_toc_parts']`` for use within
        this method. The resulting tuple is also used to properly nest children
        within parents in the table of contents.

        An example implementations of this method is within the python domain
        (:meth:`PyObject._toc_entry_name`). The python domain sets the
        ``_toc_parts`` attribute within the :py:meth:`handle_signature()`
        method.
        """
    return ''