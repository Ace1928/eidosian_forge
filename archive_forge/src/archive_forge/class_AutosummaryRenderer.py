import argparse
import inspect
import locale
import os
import pkgutil
import pydoc
import re
import sys
from gettext import NullTranslations
from os import path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Set, Tuple, Type
from jinja2 import TemplateNotFound
from jinja2.sandbox import SandboxedEnvironment
import sphinx.locale
from sphinx import __display_version__, package_dir
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.config import Config
from sphinx.ext.autodoc import Documenter
from sphinx.ext.autodoc.importer import import_module
from sphinx.ext.autosummary import (ImportExceptionGroup, get_documenter, import_by_name,
from sphinx.locale import __
from sphinx.pycode import ModuleAnalyzer, PycodeError
from sphinx.registry import SphinxComponentRegistry
from sphinx.util import logging, rst, split_full_qualified_name
from sphinx.util.inspect import getall, safe_getattr
from sphinx.util.osutil import ensuredir
from sphinx.util.template import SphinxTemplateLoader
class AutosummaryRenderer:
    """A helper class for rendering."""

    def __init__(self, app: Sphinx) -> None:
        if isinstance(app, Builder):
            raise ValueError('Expected a Sphinx application object!')
        system_templates_path = [os.path.join(package_dir, 'ext', 'autosummary', 'templates')]
        loader = SphinxTemplateLoader(app.srcdir, app.config.templates_path, system_templates_path)
        self.env = SandboxedEnvironment(loader=loader)
        self.env.filters['escape'] = rst.escape
        self.env.filters['e'] = rst.escape
        self.env.filters['underline'] = _underline
        if app.translator:
            self.env.add_extension('jinja2.ext.i18n')
            self.env.install_gettext_translations(app.translator)

    def render(self, template_name: str, context: Dict) -> str:
        """Render a template file."""
        try:
            template = self.env.get_template(template_name)
        except TemplateNotFound:
            try:
                template = self.env.get_template('autosummary/%s.rst' % template_name)
            except TemplateNotFound:
                template = self.env.get_template('autosummary/base.rst')
        return template.render(context)