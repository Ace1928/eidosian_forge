import os
import pickle
import sys
import warnings
from collections import deque
from io import StringIO
from os import path
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union
from docutils import nodes
from docutils.nodes import Element, TextElement
from docutils.parsers import Parser
from docutils.parsers.rst import Directive, roles
from docutils.transforms import Transform
from pygments.lexer import Lexer
import sphinx
from sphinx import locale, package_dir
from sphinx.config import Config
from sphinx.deprecation import RemovedInSphinx60Warning
from sphinx.domains import Domain, Index
from sphinx.environment import BuildEnvironment
from sphinx.environment.collectors import EnvironmentCollector
from sphinx.errors import ApplicationError, ConfigError, VersionRequirementError
from sphinx.events import EventManager
from sphinx.extension import Extension
from sphinx.highlighting import lexer_classes
from sphinx.locale import __
from sphinx.project import Project
from sphinx.registry import SphinxComponentRegistry
from sphinx.roles import XRefRole
from sphinx.theming import Theme
from sphinx.util import docutils, logging, progress_message
from sphinx.util.build_phase import BuildPhase
from sphinx.util.console import bold  # type: ignore
from sphinx.util.i18n import CatalogRepository
from sphinx.util.logging import prefixed_warnings
from sphinx.util.osutil import abspath, ensuredir, relpath
from sphinx.util.tags import Tags
from sphinx.util.typing import RoleFunction, TitleGetter
def add_directive(self, name: str, cls: Type[Directive], override: bool=False) -> None:
    """Register a Docutils directive.

        :param name: The name of the directive
        :param cls: A directive class
        :param override: If false, do not install it if another directive
                         is already installed as the same name
                         If true, unconditionally install the directive.

        For example, a custom directive named ``my-directive`` would be added
        like this:

        .. code-block:: python

           from docutils.parsers.rst import Directive, directives

           class MyDirective(Directive):
               has_content = True
               required_arguments = 1
               optional_arguments = 0
               final_argument_whitespace = True
               option_spec = {
                   'class': directives.class_option,
                   'name': directives.unchanged,
               }

               def run(self):
                   ...

           def setup(app):
               app.add_directive('my-directive', MyDirective)

        For more details, see `the Docutils docs
        <https://docutils.sourceforge.io/docs/howto/rst-directives.html>`__ .

        .. versionchanged:: 0.6
           Docutils 0.5-style directive classes are now supported.
        .. deprecated:: 1.8
           Docutils 0.4-style (function based) directives support is deprecated.
        .. versionchanged:: 1.8
           Add *override* keyword.
        """
    logger.debug('[app] adding directive: %r', (name, cls))
    if not override and docutils.is_directive_registered(name):
        logger.warning(__('directive %r is already registered, it will be overridden'), name, type='app', subtype='add_directive')
    docutils.register_directive(name, cls)