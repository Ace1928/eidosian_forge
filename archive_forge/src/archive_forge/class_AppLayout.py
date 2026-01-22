import re
from collections import defaultdict
from traitlets import Instance, Bool, Unicode, CUnicode, CaselessStrEnum, Tuple
from traitlets import Integer
from traitlets import HasTraits, TraitError
from traitlets import observe, validate
from .widget import Widget
from .widget_box import GridBox
from .docutils import doc_subst
@doc_subst(_doc_snippets)
class AppLayout(GridBox, LayoutProperties):
    """ Define an application like layout of widgets.

    Parameters
    ----------

    header: instance of Widget
    left_sidebar: instance of Widget
    center: instance of Widget
    right_sidebar: instance of Widget
    footer: instance of Widget
        widgets to fill the positions in the layout

    merge: bool
        flag to say whether the empty positions should be automatically merged

    pane_widths: list of numbers/strings
        the fraction of the total layout width each of the central panes should occupy
        (left_sidebar,
        center, right_sidebar)

    pane_heights: list of numbers/strings
        the fraction of the width the vertical space that the panes should occupy
         (left_sidebar, center, right_sidebar)

    {style_params}

    Examples
    --------

    """
    header = Instance(Widget, allow_none=True)
    footer = Instance(Widget, allow_none=True)
    left_sidebar = Instance(Widget, allow_none=True)
    right_sidebar = Instance(Widget, allow_none=True)
    center = Instance(Widget, allow_none=True)
    pane_widths = Tuple(CUnicode(), CUnicode(), CUnicode(), default_value=['1fr', '2fr', '1fr'])
    pane_heights = Tuple(CUnicode(), CUnicode(), CUnicode(), default_value=['1fr', '3fr', '1fr'])
    merge = Bool(default_value=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._update_layout()

    @staticmethod
    def _size_to_css(size):
        if re.match('\\d+\\.?\\d*(px|fr|%)$', size):
            return size
        if re.match('\\d+\\.?\\d*$', size):
            return size + 'fr'
        raise TypeError("the pane sizes must be in one of the following formats: '10px', '10fr', 10 (will be converted to '10fr').Got '{}'".format(size))

    def _convert_sizes(self, size_list):
        return list(map(self._size_to_css, size_list))

    def _update_layout(self):
        grid_template_areas = [['header', 'header', 'header'], ['left-sidebar', 'center', 'right-sidebar'], ['footer', 'footer', 'footer']]
        grid_template_columns = self._convert_sizes(self.pane_widths)
        grid_template_rows = self._convert_sizes(self.pane_heights)
        all_children = {'header': self.header, 'footer': self.footer, 'left-sidebar': self.left_sidebar, 'right-sidebar': self.right_sidebar, 'center': self.center}
        children = {position: child for position, child in all_children.items() if child is not None}
        if not children:
            return
        for position, child in children.items():
            child.layout.grid_area = position
        if self.merge:
            if len(children) == 1:
                position = list(children.keys())[0]
                grid_template_areas = [[position, position, position], [position, position, position], [position, position, position]]
            else:
                if self.center is None:
                    for row in grid_template_areas:
                        del row[1]
                    del grid_template_columns[1]
                if self.left_sidebar is None:
                    grid_template_areas[1][0] = grid_template_areas[1][1]
                if self.right_sidebar is None:
                    grid_template_areas[1][-1] = grid_template_areas[1][-2]
                if self.left_sidebar is None and self.right_sidebar is None and (self.center is None):
                    grid_template_areas = [['header'], ['footer']]
                    grid_template_columns = ['1fr']
                    grid_template_rows = ['1fr', '1fr']
                if self.header is None:
                    del grid_template_areas[0]
                    del grid_template_rows[0]
                if self.footer is None:
                    del grid_template_areas[-1]
                    del grid_template_rows[-1]
        grid_template_areas_css = '\n'.join(('"{}"'.format(' '.join(line)) for line in grid_template_areas))
        self.layout.grid_template_columns = ' '.join(grid_template_columns)
        self.layout.grid_template_rows = ' '.join(grid_template_rows)
        self.layout.grid_template_areas = grid_template_areas_css
        self.children = tuple(children.values())

    @observe('footer', 'header', 'center', 'left_sidebar', 'right_sidebar', 'merge', 'pane_widths', 'pane_heights')
    def _child_changed(self, change):
        self._update_layout()