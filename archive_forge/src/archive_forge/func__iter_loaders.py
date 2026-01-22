from __future__ import annotations
import typing as t
from jinja2 import BaseLoader
from jinja2 import Environment as BaseEnvironment
from jinja2 import Template
from jinja2 import TemplateNotFound
from .globals import _cv_app
from .globals import _cv_request
from .globals import current_app
from .globals import request
from .helpers import stream_with_context
from .signals import before_render_template
from .signals import template_rendered
def _iter_loaders(self, template: str) -> t.Iterator[tuple[Scaffold, BaseLoader]]:
    loader = self.app.jinja_loader
    if loader is not None:
        yield (self.app, loader)
    for blueprint in self.app.iter_blueprints():
        loader = blueprint.jinja_loader
        if loader is not None:
            yield (blueprint, loader)