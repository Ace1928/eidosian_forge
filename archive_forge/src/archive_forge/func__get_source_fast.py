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
def _get_source_fast(self, environment: BaseEnvironment, template: str) -> tuple[str, str | None, t.Callable[[], bool] | None]:
    for _srcobj, loader in self._iter_loaders(template):
        try:
            return loader.get_source(environment, template)
        except TemplateNotFound:
            continue
    raise TemplateNotFound(template)