import os
import re
import warnings
from contextlib import contextmanager
from copy import copy
from os import path
from types import ModuleType
from typing import (IO, TYPE_CHECKING, Any, Callable, Dict, Generator, List, Optional, Set,
import docutils
from docutils import nodes
from docutils.io import FileOutput
from docutils.nodes import Element, Node, system_message
from docutils.parsers.rst import Directive, directives, roles
from docutils.parsers.rst.states import Inliner
from docutils.statemachine import State, StateMachine, StringList
from docutils.utils import Reporter, unescape
from docutils.writers._html_base import HTMLTranslator
from sphinx.deprecation import RemovedInSphinx70Warning, deprecated_alias
from sphinx.errors import SphinxError
from sphinx.locale import _, __
from sphinx.util import logging
from sphinx.util.typing import RoleFunction
class SphinxTranslator(nodes.NodeVisitor):
    """A base class for Sphinx translators.

    This class adds a support for visitor/departure method for super node class
    if visitor/departure method for node class is not found.

    It also provides helper methods for Sphinx translators.

    .. note:: The subclasses of this class might not work with docutils.
              This class is strongly coupled with Sphinx.
    """

    def __init__(self, document: nodes.document, builder: 'Builder') -> None:
        super().__init__(document)
        self.builder = builder
        self.config = builder.config
        self.settings = document.settings

    def dispatch_visit(self, node: Node) -> None:
        """
        Dispatch node to appropriate visitor method.
        The priority of visitor method is:

        1. ``self.visit_{node_class}()``
        2. ``self.visit_{super_node_class}()``
        3. ``self.unknown_visit()``
        """
        for node_class in node.__class__.__mro__:
            method = getattr(self, 'visit_%s' % node_class.__name__, None)
            if method:
                method(node)
                break
        else:
            super().dispatch_visit(node)

    def dispatch_departure(self, node: Node) -> None:
        """
        Dispatch node to appropriate departure method.
        The priority of departure method is:

        1. ``self.depart_{node_class}()``
        2. ``self.depart_{super_node_class}()``
        3. ``self.unknown_departure()``
        """
        for node_class in node.__class__.__mro__:
            method = getattr(self, 'depart_%s' % node_class.__name__, None)
            if method:
                method(node)
                break
        else:
            super().dispatch_departure(node)

    def unknown_visit(self, node: Node) -> None:
        logger.warning(__('unknown node type: %r'), node, location=node)