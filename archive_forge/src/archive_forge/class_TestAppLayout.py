from unittest import TestCase
from unittest import mock
import pytest
import traitlets
import ipywidgets as widgets
from ipywidgets.widgets.widget_templates import LayoutProperties
class TestAppLayout(TestCase):
    """test layout templates"""

    def test_create_with_defaults(self):
        """test creating with default values"""
        footer = widgets.Button()
        header = widgets.Button()
        center = widgets.Button()
        left_sidebar = widgets.Button()
        right_sidebar = widgets.Button()
        box = widgets.AppLayout(footer=footer, header=header, center=center, left_sidebar=left_sidebar, right_sidebar=right_sidebar)
        assert box.layout.grid_template_areas == '"header header header"\n' + '"left-sidebar center right-sidebar"\n' + '"footer footer footer"'
        assert box.footer.layout.grid_area == 'footer'
        assert box.header.layout.grid_area == 'header'
        assert box.center.layout.grid_area == 'center'
        assert box.left_sidebar.layout.grid_area == 'left-sidebar'
        assert box.right_sidebar.layout.grid_area == 'right-sidebar'
        assert len(box.get_state()['children']) == 5
        box = widgets.AppLayout()
        assert box.layout.grid_template_areas is None
        assert box.layout.grid_template_columns is None
        assert box.layout.grid_template_rows is None
        assert len(box.get_state()['children']) == 0

    def test_merge_empty_cells(self):
        """test if cells are correctly merged"""
        footer = widgets.Button()
        header = widgets.Button()
        center = widgets.Button()
        left_sidebar = widgets.Button()
        right_sidebar = widgets.Button()
        box = widgets.AppLayout(center=center)
        assert box.layout.grid_template_areas == '"center center center"\n' + '"center center center"\n' + '"center center center"'
        assert box.center.layout.grid_area == 'center'
        assert len(box.get_state()['children']) == 1
        box = widgets.AppLayout(left_sidebar=left_sidebar)
        assert box.layout.grid_template_areas == '"left-sidebar left-sidebar left-sidebar"\n' + '"left-sidebar left-sidebar left-sidebar"\n' + '"left-sidebar left-sidebar left-sidebar"'
        assert box.left_sidebar.layout.grid_area == 'left-sidebar'
        assert len(box.get_state()['children']) == 1
        box = widgets.AppLayout(header=header, footer=footer, left_sidebar=left_sidebar, center=center)
        assert box.layout.grid_template_areas == '"header header header"\n' + '"left-sidebar center center"\n' + '"footer footer footer"'
        assert box.footer.layout.grid_area == 'footer'
        assert box.header.layout.grid_area == 'header'
        assert box.center.layout.grid_area == 'center'
        assert box.left_sidebar.layout.grid_area == 'left-sidebar'
        assert len(box.get_state()['children']) == 4
        box = widgets.AppLayout(header=header, footer=footer, right_sidebar=right_sidebar, center=center)
        assert box.layout.grid_template_areas == '"header header header"\n' + '"center center right-sidebar"\n' + '"footer footer footer"'
        assert box.footer.layout.grid_area == 'footer'
        assert box.header.layout.grid_area == 'header'
        assert box.center.layout.grid_area == 'center'
        assert box.right_sidebar.layout.grid_area == 'right-sidebar'
        assert len(box.get_state()['children']) == 4
        box = widgets.AppLayout(header=header, footer=footer, center=center)
        assert box.layout.grid_template_areas == '"header header header"\n' + '"center center center"\n' + '"footer footer footer"'
        assert box.footer.layout.grid_area == 'footer'
        assert box.header.layout.grid_area == 'header'
        assert box.center.layout.grid_area == 'center'
        assert len(box.get_state()['children']) == 3
        box = widgets.AppLayout(header=header, footer=footer, center=None, left_sidebar=left_sidebar, right_sidebar=right_sidebar)
        assert box.layout.grid_template_areas == '"header header"\n' + '"left-sidebar right-sidebar"\n' + '"footer footer"'
        assert box.footer.layout.grid_area == 'footer'
        assert box.header.layout.grid_area == 'header'
        assert box.left_sidebar.layout.grid_area == 'left-sidebar'
        assert box.right_sidebar.layout.grid_area == 'right-sidebar'
        assert box.center is None
        assert len(box.get_state()['children']) == 4
        box = widgets.AppLayout(header=header, footer=footer, center=None, left_sidebar=None, right_sidebar=right_sidebar)
        assert box.layout.grid_template_areas == '"header header"\n' + '"right-sidebar right-sidebar"\n' + '"footer footer"'
        assert box.footer.layout.grid_area == 'footer'
        assert box.header.layout.grid_area == 'header'
        assert box.left_sidebar is None
        assert box.right_sidebar.layout.grid_area == 'right-sidebar'
        assert box.center is None
        assert len(box.get_state()['children']) == 3
        box = widgets.AppLayout(header=header, footer=footer, center=None, left_sidebar=None, right_sidebar=None)
        assert box.layout.grid_template_areas == '"header"\n' + '"footer"'
        assert box.footer.layout.grid_area == 'footer'
        assert box.header.layout.grid_area == 'header'
        assert box.center is None
        assert box.left_sidebar is None
        assert box.right_sidebar is None
        assert len(box.get_state()['children']) == 2
        box = widgets.AppLayout(header=header, footer=footer, center=center, merge=False)
        assert box.layout.grid_template_areas == '"header header header"\n' + '"left-sidebar center right-sidebar"\n' + '"footer footer footer"'
        assert box.footer.layout.grid_area == 'footer'
        assert box.header.layout.grid_area == 'header'
        assert box.center.layout.grid_area == 'center'
        assert box.left_sidebar is None
        assert box.right_sidebar is None
        assert len(box.get_state()['children']) == 3
        box = widgets.AppLayout(footer=footer, center=center, left_sidebar=left_sidebar, right_sidebar=right_sidebar)
        assert box.layout.grid_template_areas == '"left-sidebar center right-sidebar"\n' + '"footer footer footer"'
        assert box.center.layout.grid_area == 'center'
        assert box.left_sidebar.layout.grid_area == 'left-sidebar'
        assert box.right_sidebar.layout.grid_area == 'right-sidebar'
        assert box.footer.layout.grid_area == 'footer'
        assert box.header is None
        assert len(box.get_state()['children']) == 4
        box = widgets.AppLayout(header=header, center=center, left_sidebar=left_sidebar, right_sidebar=right_sidebar)
        assert box.layout.grid_template_areas == '"header header header"\n' + '"left-sidebar center right-sidebar"'
        assert box.center.layout.grid_area == 'center'
        assert box.left_sidebar.layout.grid_area == 'left-sidebar'
        assert box.right_sidebar.layout.grid_area == 'right-sidebar'
        assert box.header.layout.grid_area == 'header'
        assert box.footer is None
        assert len(box.get_state()['children']) == 4
        box = widgets.AppLayout(center=center, left_sidebar=left_sidebar, right_sidebar=right_sidebar)
        assert box.layout.grid_template_areas == '"left-sidebar center right-sidebar"'
        assert box.center.layout.grid_area == 'center'
        assert box.left_sidebar.layout.grid_area == 'left-sidebar'
        assert box.right_sidebar.layout.grid_area == 'right-sidebar'
        assert box.footer is None
        assert box.header is None
        assert len(box.get_state()['children']) == 3
        box = widgets.AppLayout(center=center)
        assert box.layout.grid_template_areas == '"center center center"\n' + '"center center center"\n' + '"center center center"'
        assert box.center.layout.grid_area == 'center'
        assert len(box.get_state()['children']) == 1

    def test_size_to_css(self):
        box = widgets.AppLayout()
        assert box._size_to_css('100px') == '100px'
        assert box._size_to_css('1fr') == '1fr'
        assert box._size_to_css('2.5fr') == '2.5fr'
        assert box._size_to_css('2.5') == '2.5fr'
        assert box._size_to_css('25%') == '25%'
        with pytest.raises(TypeError):
            box._size_to_css('this is not correct size')

    def test_set_pane_widths_heights(self):
        footer = widgets.Button()
        header = widgets.Button()
        center = widgets.Button()
        left_sidebar = widgets.Button()
        right_sidebar = widgets.Button()
        box = widgets.AppLayout(header=header, footer=footer, left_sidebar=left_sidebar, right_sidebar=left_sidebar, center=center)
        with pytest.raises(traitlets.TraitError):
            box.pane_widths = ['1fx', '1fx', '1fx', '1fx']
        with pytest.raises(traitlets.TraitError):
            box.pane_widths = ['1fx', '1fx']
        with pytest.raises(traitlets.TraitError):
            box.pane_heights = ['1fx', '1fx', '1fx', '1fx']
        with pytest.raises(traitlets.TraitError):
            box.pane_heights = ['1fx', '1fx']
        assert box.layout.grid_template_rows == '1fr 3fr 1fr'
        assert box.layout.grid_template_columns == '1fr 2fr 1fr'
        box.pane_heights = ['3fr', '100px', 20]
        assert box.layout.grid_template_rows == '3fr 100px 20fr'
        assert box.layout.grid_template_columns == '1fr 2fr 1fr'
        box.pane_widths = [3, 3, 1]
        assert box.layout.grid_template_rows == '3fr 100px 20fr'
        assert box.layout.grid_template_columns == '3fr 3fr 1fr'