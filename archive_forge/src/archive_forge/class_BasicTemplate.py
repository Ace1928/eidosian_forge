from __future__ import annotations
import os
import sys
import uuid
from functools import partial
from pathlib import Path, PurePath
from typing import (
import jinja2
import param
from bokeh.document.document import Document
from bokeh.models import LayoutDOM
from bokeh.settings import settings as _settings
from pyviz_comms import JupyterCommManager as _JupyterCommManager
from ..config import _base_config, config, panel_extension
from ..io.document import init_doc
from ..io.model import add_to_doc
from ..io.notebook import render_template
from ..io.notifications import NotificationArea
from ..io.resources import (
from ..io.save import save
from ..io.state import curdoc_locked, state
from ..layout import Column, GridSpec, ListLike
from ..models.comm_manager import CommManager
from ..pane import (
from ..pane.image import ImageBase
from ..reactive import ReactiveHTML
from ..theme.base import (
from ..util import isurl
from ..viewable import (
from ..widgets import Button
from ..widgets.indicators import BooleanIndicator, LoadingSpinner
class BasicTemplate(BaseTemplate):
    """
    BasicTemplate provides a baseclass for templates with a basic
    organization including a header, sidebar and main area. Unlike the
    more generic Template class these default templates make it easy
    for a user to generate an application with a polished look and
    feel without having to write any Jinja2 template themselves.
    """
    busy_indicator = param.ClassSelector(default=LoadingSpinner(width=20, height=20), class_=BooleanIndicator, constant=True, allow_None=True, doc='\n        Visual indicator of application busy state.')
    collapsed_sidebar = param.Selector(default=False, constant=True, doc='\n        Whether the sidebar (if present) is initially collapsed.')
    header = param.ClassSelector(class_=ListLike, constant=True, doc='\n        A list-like container which populates the header bar.')
    main = param.ClassSelector(class_=ListLike, constant=True, doc='\n        A list-like container which populates the main area.')
    main_max_width = param.String(default='', doc="\n        The maximum width of the main area. For example '800px' or '80%'.\n        If the string is '' (default) no max width is set.")
    sidebar = param.ClassSelector(class_=ListLike, constant=True, doc='\n        A list-like container which populates the sidebar.')
    sidebar_width = param.Integer(default=330, doc='\n        The width of the sidebar in pixels. Default is 330.')
    modal = param.ClassSelector(class_=ListLike, constant=True, doc='\n        A list-like container which populates the modal')
    notifications = param.ClassSelector(class_=NotificationArea, constant=True, doc='\n        The NotificationArea instance attached to this template.\n        Automatically added if config.notifications is set, but may\n        also be provided explicitly.')
    logo = param.String(doc="\n        URI of logo to add to the header (if local file, logo is\n        base64 encoded as URI). Default is '', i.e. not shown.")
    favicon = param.String(default=FAVICON_URL, doc='\n        URI of favicon to add to the document head (if local file, favicon is\n        base64 encoded as URI).')
    title = param.String(default='Panel Application', doc='\n        A title to show in the header. Also added to the document head\n        meta settings and as the browser tab title.')
    site = param.String(default='', doc="\n        Name of the site. Will be shown in the header and link to the\n        'site_url'. Default is '', i.e. not shown.")
    site_url = param.String(default='/', doc="\n        Url of the site and logo. Default is '/'.")
    manifest = param.String(default=None, doc='\n        Manifest to add to site.')
    meta_description = param.String(doc="\n        A meta description to add to the document head for search\n        engine optimization. For example 'P.A. Nelson'.")
    meta_keywords = param.String(doc='\n        Meta keywords to add to the document head for search engine\n        optimization.')
    meta_author = param.String(doc="\n        A meta author to add to the the document head for search\n        engine optimization. For example 'P.A. Nelson'.")
    meta_refresh = param.String(doc="\n        A meta refresh rate to add to the document head. For example\n        '30' will instruct the browser to refresh every 30\n        seconds. Default is '', i.e. no automatic refresh.")
    meta_viewport = param.String(doc='\n        A meta viewport to add to the header.')
    base_url = param.String(doc="\n        Specifies the base URL for all relative URLs in a\n        page. Default is '', i.e. not the domain.")
    base_target = param.ObjectSelector(default='_self', objects=['_blank', '_self', '_parent', '_top'], doc='\n        Specifies the base Target for all relative URLs in a page.')
    header_background = param.String(doc='\n        Optional header background color override.')
    header_color = param.String(doc='\n        Optional header text color override.')
    location = param.Boolean(default=True, readonly=True)
    _actions = param.ClassSelector(default=TemplateActions(), class_=TemplateActions)
    _template: ClassVar[Path | None] = None
    __abstract = True

    def __init__(self, **params):
        tmpl_string = self._template.read_text(encoding='utf-8')
        try:
            template = _env.get_template(str(self._template.relative_to(Path(__file__).parent)))
        except (jinja2.exceptions.TemplateNotFound, ValueError):
            template = parse_template(tmpl_string)
        if 'header' not in params:
            params['header'] = ListLike()
        else:
            params['header'] = self._get_params(params['header'], self.param.header.class_)
        if 'main' not in params:
            params['main'] = ListLike()
        else:
            params['main'] = self._get_params(params['main'], self.param.main.class_)
        if 'sidebar' not in params:
            params['sidebar'] = ListLike()
        else:
            params['sidebar'] = self._get_params(params['sidebar'], self.param.sidebar.class_)
        if 'modal' not in params:
            params['modal'] = ListLike()
        else:
            params['modal'] = self._get_params(params['modal'], self.param.modal.class_)
        if 'theme' in params:
            if isinstance(params['theme'], str):
                params['theme'] = THEMES[params['theme']]
        else:
            params['theme'] = THEMES[config.theme]
        if 'favicon' in params and isinstance(params['favicon'], PurePath):
            params['favicon'] = str(params['favicon'])
        if 'notifications' not in params and config.notifications:
            params['notifications'] = state.notifications if state.curdoc else NotificationArea()
        super().__init__(template=template, **params)
        self._js_area = HTML(margin=0, width=0, height=0)
        state_roots = '{% block state_roots %}' in tmpl_string
        if state_roots or 'embed(roots.js_area)' in tmpl_string:
            self._render_items['js_area'] = (self._js_area, [])
        if state_roots or 'embed(roots.actions)' in tmpl_string:
            self._render_items['actions'] = (self._actions, [])
        if (state_roots or 'embed(roots.notifications)' in tmpl_string) and self.notifications:
            self._render_items['notifications'] = (self.notifications, [])
            self._render_variables['notifications'] = True
        if config.browser_info and ('embed(roots.browser_info)' in tmpl_string or state_roots) and state.browser_info:
            self._render_items['browser_info'] = (state.browser_info, [])
            self._render_variables['browser_info'] = True
        self._update_busy()
        self.main.param.watch(self._update_render_items, ['objects'])
        self.modal.param.watch(self._update_render_items, ['objects'])
        self.sidebar.param.watch(self._update_render_items, ['objects'])
        self.header.param.watch(self._update_render_items, ['objects'])
        self.main.param.trigger('objects')
        self.sidebar.param.trigger('objects')
        self.header.param.trigger('objects')
        self.modal.param.trigger('objects')

    def _init_doc(self, doc: Optional[Document]=None, comm: Optional['Comm']=None, title: Optional[str]=None, notebook: bool=False, location: bool | Location=True) -> Document:
        title = self.title if self.title != self.param.title.default else title
        if self.busy_indicator:
            state.sync_busy(self.busy_indicator)
        document = super()._init_doc(doc, comm, title, notebook, location)
        if self.notifications:
            state._notifications[document] = self.notifications
        if self._design.theme.bokeh_theme:
            document.theme = self._design.theme.bokeh_theme
        return document

    def _update_vars(self, *args) -> None:
        super()._update_vars(*args)
        self._render_variables['app_title'] = self.title
        self._render_variables['meta_name'] = self.title
        self._render_variables['site_title'] = self.site
        self._render_variables['site_url'] = self.site_url
        self._render_variables['manifest'] = self.manifest
        self._render_variables['meta_description'] = self.meta_description
        self._render_variables['meta_keywords'] = self.meta_keywords
        self._render_variables['meta_author'] = self.meta_author
        self._render_variables['meta_refresh'] = self.meta_refresh
        self._render_variables['meta_viewport'] = self.meta_viewport
        self._render_variables['base_url'] = self.base_url
        self._render_variables['base_target'] = self.base_target
        if os.path.isfile(self.logo):
            img = _panel(self.logo)
            if not isinstance(img, ImageBase):
                raise ValueError(f'Could not determine file type of logo: {self.logo}.')
            logo = img._b64(img._data(img.object))
        else:
            logo = self.logo
        if os.path.isfile(self.favicon):
            img = _panel(self.favicon)
            if not isinstance(img, ImageBase):
                raise ValueError(f'Could not determine file type of favicon: {self.favicon}.')
            favicon = img._b64(img._data(img.object))
        elif _settings.resources(default='server') == 'cdn' and self.favicon == FAVICON_URL:
            favicon = CDN_DIST + 'images/favicon.ico'
        else:
            favicon = self.favicon
        self._render_variables['app_logo'] = logo
        self._render_variables['app_favicon'] = favicon
        self._render_variables['app_favicon_type'] = self._get_favicon_type(self.favicon)
        self._render_variables['header_background'] = self.header_background
        self._render_variables['header_color'] = self.header_color
        self._render_variables['main_max_width'] = self.main_max_width
        self._render_variables['sidebar_width'] = self.sidebar_width
        self._render_variables['theme'] = self._design.theme
        self._render_variables['collapsed_sidebar'] = self.collapsed_sidebar

    def _update_busy(self) -> None:
        if self.busy_indicator:
            self._render_items['busy_indicator'] = (self.busy_indicator, [])
        elif 'busy_indicator' in self._render_items:
            del self._render_items['busy_indicator']
        self._render_variables['busy'] = self.busy_indicator is not None

    def _update_render_items(self, event: param.parameterized.Event) -> None:
        if event.obj is self and event.name == 'busy_indicator':
            return self._update_busy()
        if event.obj is self.main:
            tag = 'main'
        elif event.obj is self.sidebar:
            tag = 'nav'
        elif event.obj is self.header:
            tag = 'header'
        elif event.obj is self.modal:
            tag = 'modal'
        old = event.old if isinstance(event.old, list) else list(event.old.values())
        for obj in old:
            ref = f'{tag}-{str(id(obj))}'
            if ref in self._render_items:
                del self._render_items[ref]
        new = event.new if isinstance(event.new, list) else event.new.values()
        if self._design.theme.bokeh_theme:
            for o in new:
                if o in old:
                    continue
                for hvpane in o.select(HoloViews):
                    hvpane.theme = self._design.theme.bokeh_theme
        labels = {}
        for obj in new:
            ref = f'{tag}-{str(id(obj))}'
            if obj.name.startswith(type(obj).__name__):
                labels[ref] = 'Content'
            else:
                labels[ref] = obj.name
            self._render_items[ref] = (obj, [tag])
        tags = [tags for _, tags in self._render_items.values()]
        self._render_variables['nav'] = any(('nav' in ts for ts in tags))
        self._render_variables['header'] = any(('header' in ts for ts in tags))
        self._render_variables['root_labels'] = labels

    def _server_destroy(self, session_context: BokehSessionContext):
        super()._server_destroy(session_context)
        if not self._documents and self.busy_indicator in state._indicators:
            state._indicators.remove(self.busy_indicator)

    def open_modal(self) -> None:
        """
        Opens the modal area
        """
        self._actions.open_modal += 1

    def close_modal(self) -> None:
        """
        Closes the modal area
        """
        self._actions.close_modal += 1

    @staticmethod
    def _get_favicon_type(favicon) -> str:
        if not favicon:
            return ''
        elif favicon.endswith('.png'):
            return 'image/png'
        elif favicon.endswith('jpg'):
            return 'image/jpg'
        elif favicon.endswith('gif'):
            return 'image/gif'
        elif favicon.endswith('svg'):
            return 'image/svg'
        elif favicon.endswith('ico'):
            return 'image/x-icon'
        else:
            raise ValueError('favicon type not supported.')

    @staticmethod
    def _get_params(value, class_):
        if isinstance(value, class_):
            return value
        if isinstance(value, tuple):
            value = [*value]
        elif not isinstance(value, list):
            value = [value]
        value = [_panel(item) for item in value]
        if class_ is ListLike:
            return ListLike(objects=value)
        if class_ is GridSpec:
            grid = GridSpec(ncols=12, mode='override')
            for index, item in enumerate(value):
                grid[index, :] = item
            return grid
        return value