import inspect
import os
import posixpath
import re
import sys
import warnings
from inspect import Parameter
from os import path
from types import ModuleType
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, cast
from docutils import nodes
from docutils.nodes import Node, system_message
from docutils.parsers.rst import directives
from docutils.parsers.rst.states import RSTStateMachine, Struct, state_classes
from docutils.statemachine import StringList
import sphinx
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.config import Config
from sphinx.deprecation import (RemovedInSphinx60Warning, RemovedInSphinx70Warning,
from sphinx.environment import BuildEnvironment
from sphinx.ext.autodoc import INSTANCEATTR, Documenter
from sphinx.ext.autodoc.directive import DocumenterBridge, Options
from sphinx.ext.autodoc.importer import import_module
from sphinx.ext.autodoc.mock import mock
from sphinx.extension import Extension
from sphinx.locale import __
from sphinx.project import Project
from sphinx.pycode import ModuleAnalyzer, PycodeError
from sphinx.registry import SphinxComponentRegistry
from sphinx.util import logging, rst
from sphinx.util.docutils import (NullReporter, SphinxDirective, SphinxRole, new_document,
from sphinx.util.inspect import signature_from_str
from sphinx.util.matching import Matcher
from sphinx.util.typing import OptionSpec
from sphinx.writers.html import HTMLTranslator
class AutoLink(SphinxRole):
    """Smart linking role.

    Expands to ':obj:`text`' if `text` is an object that can be imported;
    otherwise expands to '*text*'.
    """

    def run(self) -> Tuple[List[Node], List[system_message]]:
        pyobj_role = self.env.get_domain('py').role('obj')
        objects, errors = pyobj_role('obj', self.rawtext, self.text, self.lineno, self.inliner, self.options, self.content)
        if errors:
            return (objects, errors)
        assert len(objects) == 1
        pending_xref = cast(addnodes.pending_xref, objects[0])
        try:
            prefixes = get_import_prefixes_from_env(self.env)
            import_by_name(pending_xref['reftarget'], prefixes)
        except ImportExceptionGroup:
            literal = cast(nodes.literal, pending_xref[0])
            objects[0] = nodes.emphasis(self.rawtext, literal.astext(), classes=literal['classes'])
        return (objects, errors)