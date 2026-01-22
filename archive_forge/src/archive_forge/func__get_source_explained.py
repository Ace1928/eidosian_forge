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
def _get_source_explained(self, environment: BaseEnvironment, template: str) -> tuple[str, str | None, t.Callable[[], bool] | None]:
    attempts = []
    rv: tuple[str, str | None, t.Callable[[], bool] | None] | None
    trv: None | tuple[str, str | None, t.Callable[[], bool] | None] = None
    for srcobj, loader in self._iter_loaders(template):
        try:
            rv = loader.get_source(environment, template)
            if trv is None:
                trv = rv
        except TemplateNotFound:
            rv = None
        attempts.append((loader, srcobj, rv))
    from .debughelpers import explain_template_loading_attempts
    explain_template_loading_attempts(self.app, template, attempts)
    if trv is not None:
        return trv
    raise TemplateNotFound(template)