from __future__ import annotations
from typing import (
import param
from bokeh.events import ButtonClick, MenuItemClick
from bokeh.models import Dropdown as _BkDropdown, Toggle as _BkToggle
from bokeh.models.ui import SVGIcon, TablerIcon
from ..io.resources import CDN_DIST
from ..links import Callback
from ..models.widgets import Button as _BkButton
from ._mixin import TooltipMixin
from .base import Widget
class MenuButton(_ButtonBase, _ClickButton, IconMixin):
    """
    The `MenuButton` widget allows specifying a list of menu items to
    select from triggering events when the button is clicked.

    Unlike other widgets, it does not have a `value`
    parameter. Instead it has a `clicked` parameter that can be
    watched to trigger events and which reports the last clicked menu
    item.

    Reference: https://panel.holoviz.org/reference/widgets/MenuButton.html

    :Example:

    >>> menu_items = [('Option A', 'a'), ('Option B', 'b'), None, ('Option C', 'c')]
    >>> MenuButton(name='Dropdown', items=menu_items, button_type='primary')
    """
    clicked = param.String(default=None, doc='\n      Last menu item that was clicked.')
    items = param.List(default=[], doc='\n      Menu items in the dropdown. Allows strings, tuples of the form\n      (title, value) or Nones to separate groups of items.')
    split = param.Boolean(default=False, doc='\n      Whether to add separate dropdown area to button.')
    _event: ClassVar[str] = 'menu_item_click'
    _rename: ClassVar[Mapping[str, str | None]] = {'name': 'label', 'items': 'menu', 'clicked': None}
    _widget_type: ClassVar[Type[Model]] = _BkDropdown

    def __init__(self, **params):
        click_handler = params.pop('on_click', None)
        super().__init__(**params)
        if click_handler:
            self.on_click(click_handler)

    def _get_model(self, doc: Document, root: Optional[Model]=None, parent: Optional[Model]=None, comm: Optional[Comm]=None) -> Model:
        model = super()._get_model(doc, root, parent, comm)
        self._register_events('button_click', model=model, doc=doc, comm=comm)
        return model

    def _process_event(self, event: ButtonClick | MenuItemClick):
        if isinstance(event, MenuItemClick):
            item = event.item
        elif isinstance(event, ButtonClick):
            item = self.name
        self.clicked = item

    def on_click(self, callback: Callable[[param.parameterized.Event], None]) -> param.parameterized.Watcher:
        """
        Register a callback to be executed when the button is clicked.

        The callback is given an `Event` argument declaring the number of clicks

        Arguments
        ---------
        callback: (Callable[[param.parameterized.Event], None])
            The function to run on click events. Must accept a positional `Event` argument

        Returns
        -------
        watcher: param.Parameterized.Watcher
          A `Watcher` that executes the callback when the MenuButton is clicked.
        """
        return self.param.watch(callback, 'clicked', onlychanged=False)