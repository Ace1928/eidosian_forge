import sys
import os
import re
import warnings
import types
import unicodedata
class sidebar(Structural, Element):
    """
    Sidebars are like miniature, parallel documents that occur inside other
    documents, providing related or reference material.  A sidebar is
    typically offset by a border and "floats" to the side of the page; the
    document's main text may flow around it.  Sidebars can also be likened to
    super-footnotes; their content is outside of the flow of the document's
    main text.

    Sidebars are allowed wherever body elements (list, table, etc.) are
    allowed, but only at the top level of a section or document.  Sidebars
    cannot nest inside sidebars, topics, or body elements; you can't have a
    sidebar inside a table, list, block quote, etc.
    """