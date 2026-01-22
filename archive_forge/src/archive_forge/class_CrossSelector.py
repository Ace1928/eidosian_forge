from a list of options.
from __future__ import annotations
import itertools
import re
from types import FunctionType
from typing import (
import numpy as np
import param
from bokeh.models import PaletteSelect
from bokeh.models.widgets import (
from ..io.resources import CDN_DIST
from ..layout.base import Column, ListPanel, NamedListPanel
from ..models import (
from ..util import PARAM_NAME_PATTERN, indexOf, isIn
from ._mixin import TooltipMixin
from .base import CompositeWidget, Widget
from .button import Button, _ButtonBase
from .input import TextAreaInput, TextInput
class CrossSelector(CompositeWidget, MultiSelect):
    """
    A composite widget which allows selecting from a list of items
    by moving them between two lists. Supports filtering values by
    name to select them in bulk.

    Reference: https://panel.holoviz.org/reference/widgets/CrossSelector.html

    :Example:

    >>> CrossSelector(
    ...     name='Fruits', value=['Apple', 'Pear'],
    ...     options=['Apple', 'Banana', 'Pear', 'Strawberry']
    ... )
    """
    width = param.Integer(default=600, allow_None=True, doc='\n        The number of options shown at once (note this is the\n        only way to control the height of this widget)')
    height = param.Integer(default=200, allow_None=True, doc='\n        The number of options shown at once (note this is the\n        only way to control the height of this widget)')
    filter_fn = param.Callable(default=re.search, doc='\n        The filter function applied when querying using the text\n        fields, defaults to re.search. Function is two arguments, the\n        query or pattern and the item label.')
    size = param.Integer(default=10, doc='\n        The number of options shown at once (note this is the only way\n        to control the height of this widget)')
    definition_order = param.Integer(default=True, doc='\n       Whether to preserve definition order after filtering. Disable\n       to allow the order of selection to define the order of the\n       selected list.')

    def __init__(self, **params):
        super().__init__(**params)
        labels, values = (self.labels, self.values)
        selected = [labels[indexOf(v, values)] for v in params.get('value', []) if isIn(v, values)]
        unselected = [k for k in labels if k not in selected]
        layout = dict(sizing_mode='stretch_both', margin=0)
        self._lists = {False: MultiSelect(options=unselected, size=self.size, **layout), True: MultiSelect(options=selected, size=self.size, **layout)}
        self._lists[False].param.watch(self._update_selection, 'value')
        self._lists[True].param.watch(self._update_selection, 'value')
        self._buttons = {False: Button(name='❮❮', width=50), True: Button(name='❯❯', width=50)}
        self._buttons[False].param.watch(self._apply_selection, 'clicks')
        self._buttons[True].param.watch(self._apply_selection, 'clicks')
        self._search = {False: TextInput(placeholder='Filter available options', margin=(0, 0, 10, 0), width_policy='max'), True: TextInput(placeholder='Filter selected options', margin=(0, 0, 10, 0), width_policy='max')}
        self._search[False].param.watch(self._filter_options, 'value_input')
        self._search[True].param.watch(self._filter_options, 'value_input')
        self._placeholder = TextAreaInput(placeholder='To select an item highlight it on the left and use the arrow button to move it to the right.', disabled=True, **layout)
        right = self._lists[True] if self.value else self._placeholder
        self._unselected = Column(self._search[False], self._lists[False], **layout)
        self._selected = Column(self._search[True], right, **layout)
        buttons = Column(self._buttons[True], self._buttons[False], margin=(0, 5), align='center')
        self._composite[:] = [self._unselected, buttons, self._selected]
        self._selections = {False: [], True: []}
        self._query = {False: '', True: ''}
        self._update_disabled()
        self._update_width()

    @param.depends('width', watch=True)
    def _update_width(self):
        width = int(self.width // 2.0 - 50)
        self._search[False].width = width
        self._search[True].width = width
        self._lists[False].width = width
        self._lists[True].width = width

    @param.depends('size', watch=True)
    def _update_size(self):
        self._lists[False].size = self.size
        self._lists[True].size = self.size

    @param.depends('disabled', watch=True)
    def _update_disabled(self):
        self._buttons[False].disabled = self.disabled
        self._buttons[True].disabled = self.disabled

    @param.depends('value', watch=True)
    def _update_value(self):
        labels, values = (self.labels, self.values)
        selected = [labels[indexOf(v, values)] for v in self.value if isIn(v, values)]
        unselected = [k for k in labels if k not in selected]
        self._lists[True].options = selected
        self._lists[True].value = []
        self._lists[False].options = unselected
        self._lists[False].value = []
        if len(self._lists[True].options) and self._selected[-1] is not self._lists[True]:
            self._selected[-1] = self._lists[True]
        elif not len(self._lists[True].options) and self._selected[-1] is not self._placeholder:
            self._selected[-1] = self._placeholder

    @param.depends('options', watch=True)
    def _update_options(self):
        """
        Updates the options of each of the sublists after the options
        for the whole widget are updated.
        """
        self._selections[False] = []
        self._selections[True] = []
        self._update_value()

    def _apply_filters(self):
        self._apply_query(False)
        self._apply_query(True)

    def _filter_options(self, event):
        """
        Filters unselected options based on a text query event.
        """
        selected = event.obj is self._search[True]
        self._query[selected] = event.new
        self._apply_query(selected)

    def _apply_query(self, selected):
        query = self._query[selected]
        other = self._lists[not selected].labels
        labels = self.labels
        if self.definition_order:
            options = [k for k in labels if k not in other]
        else:
            options = self._lists[selected].values
        if not query:
            self._lists[selected].options = options
            self._lists[selected].value = []
        else:
            try:
                matches = [o for o in options if self.filter_fn(query, o)]
            except Exception:
                matches = []
            self._lists[selected].options = options if options else []
            self._lists[selected].value = [m for m in matches]

    def _update_selection(self, event):
        """
        Updates the current selection in each list.
        """
        selected = event.obj is self._lists[True]
        self._selections[selected] = [v for v in event.new if v != '']

    def _apply_selection(self, event):
        """
        Applies the current selection depending on which button was
        pressed.
        """
        selected = event.obj is self._buttons[True]
        new = {k: self._items[k] for k in self._selections[not selected]}
        old = self._lists[selected].options
        other = self._lists[not selected].options
        merged = {k: k for k in list(old) + list(new)}
        leftovers = {k: k for k in other if k not in new}
        self._lists[selected].options = merged if merged else {}
        self._lists[not selected].options = leftovers if leftovers else {}
        if len(self._lists[True].options):
            self._selected[-1] = self._lists[True]
        else:
            self._selected[-1] = self._placeholder
        self.value = [self._items[o] for o in self._lists[True].options if o != '']
        self._apply_filters()

    def _get_model(self, doc, root=None, parent=None, comm=None):
        return self._composite._get_model(doc, root, parent, comm)