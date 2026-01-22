from .widget_box import Box
from .widget import register
from .widget_core import CoreWidget
from traitlets import Unicode, Dict, CInt, TraitError, validate, observe
from .trait_types import TypedTuple
from itertools import chain, repeat, islice
class _SelectionContainer(Box, CoreWidget):
    """Base class used to display multiple child widgets."""
    titles = TypedTuple(trait=Unicode(), help='Titles of the pages').tag(sync=True)
    selected_index = CInt(help='The index of the selected page. This is either an integer selecting a particular sub-widget, or None to have no widgets selected.', allow_none=True, default_value=None).tag(sync=True)

    @validate('selected_index')
    def _validated_index(self, proposal):
        if proposal.value is None or 0 <= proposal.value < len(self.children):
            return proposal.value
        else:
            raise TraitError('Invalid selection: index out of bounds')

    @validate('titles')
    def _validate_titles(self, proposal):
        return tuple(pad(proposal.value, '', len(self.children)))

    @observe('children')
    def _observe_children(self, change):
        self._reset_selected_index()
        self._reset_titles()

    def _reset_selected_index(self):
        if self.selected_index is not None and len(self.children) < self.selected_index:
            self.selected_index = None

    def _reset_titles(self):
        if len(self.titles) != len(self.children):
            self.titles = tuple(self.titles)

    def set_title(self, index, title):
        """Sets the title of a container page.
        Parameters
        ----------
        index : int
            Index of the container page
        title : unicode
            New title
        """
        titles = list(self.titles)
        if title is None:
            title = ''
        titles[index] = title
        self.titles = tuple(titles)

    def get_title(self, index):
        """Gets the title of a container page.
        Parameters
        ----------
        index : int
            Index of the container page
        """
        return self.titles[index]