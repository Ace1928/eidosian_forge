import traceback
import warnings
from importlib import import_module
from types import MethodType
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Tuple, Type,
from docutils import nodes
from docutils.core import Publisher
from docutils.io import Input
from docutils.nodes import Element, Node, TextElement
from docutils.parsers import Parser
from docutils.parsers.rst import Directive
from docutils.transforms import Transform
from sphinx.builders import Builder
from sphinx.config import Config
from sphinx.deprecation import RemovedInSphinx60Warning, RemovedInSphinx70Warning
from sphinx.domains import Domain, Index, ObjType
from sphinx.domains.std import GenericObject, Target
from sphinx.environment import BuildEnvironment
from sphinx.errors import ExtensionError, SphinxError, VersionRequirementError
from sphinx.extension import Extension
from sphinx.io import create_publisher
from sphinx.locale import __
from sphinx.parsers import Parser as SphinxParser
from sphinx.roles import XRefRole
from sphinx.util import logging
from sphinx.util.logging import prefixed_warnings
from sphinx.util.typing import RoleFunction, TitleGetter
def add_translation_handlers(self, node: Type[Element], **kwargs: Tuple[Callable, Callable]) -> None:
    logger.debug('[app] adding translation_handlers: %r, %r', node, kwargs)
    for builder_name, handlers in kwargs.items():
        translation_handlers = self.translation_handlers.setdefault(builder_name, {})
        try:
            visit, depart = handlers
            translation_handlers[node.__name__] = (visit, depart)
        except ValueError as exc:
            raise ExtensionError(__('kwargs for add_node() must be a (visit, depart) function tuple: %r=%r') % (builder_name, handlers)) from exc