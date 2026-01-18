from traitlets import Int, Unicode, List, Dict, Bool, Instance, Float
from traittypes import Array, DataFrame
from ipywidgets import (DOMWidget, CallbackDispatcher, Color,
from ipywidgets.widgets.widget_layout import LayoutTraitType
from .traits import (array_serialization, dataframe_serialization,
from .marks import CATEGORY10
from ._version import __frontend_version__
Waffle wrapped map.

    Attributes
    ----------
    names: numpy.ndarray of strings (default: [])
        The elements can also be objects convertible to string
        primary key for the map data. A rectangle is created for each
        unique entry in this array
    groups: numpy.ndarray (default: [])
        attribute on which the groupby is run. If this is an empty array, then
        there is no group by for the map.
    display_text: numpy.ndarray or None(default: None)
        data to be displayed on each rectangle of the map.If this is empty it
        defaults to the names attribute.
    ref_data: pandas.DataDrame or None (default: None)
        Additional data associated with each element of the map. The data in
        this data frame can be displayed as a tooltip.
    color: numpy.ndarray (default: [])
        Data to represent the color for each of the cells. If the value of the
        data is NaN for a cell, then the color of the cell is the color of the
        group it belongs to in absence of data for color
    scales: Dictionary of scales holding a scale for each data attribute
        If the map has data being passed as color, then a corresponding color
        scale is required
    axes: List of axes
        Ability to add an axis for the scales which are used to scale data
        represented in the map
    on_hover: custom event
        This event is received when the mouse is hovering over a cell. Returns
        the data of the cell and the ref_data associated with the cell.
    tooltip_widget: Instance of a widget
        Widget to be displayed as the tooltip. This can be combined with the
        on_hover event to display the chart corresponding to the cell being
        hovered on.
    tooltip_fields: list
        names of the fields from the ref_data dataframe which should be
        displayed in the tooltip.
    tooltip_formats: list
        formats for each of the fields for the tooltip data. Order should match
        the order of the tooltip_fields
    freeze_tooltip_location: bool (default: False)
        if True, freezes the location of the tooltip. If False, tootip will
        follow the mouse
    show_groups: bool
        attribute to determine if the groups should be displayed. If set to
        True, the finer elements are blurred

    Map Drawing Attributes
    cols: int
        Suggestion for no of columns in the map.If not specified, value is
        inferred from the no of rows and no of cells
    rows: int
        No of rows in the map.If not specified, value is inferred from the no
        of cells and no of columns.
        If both rows and columns are not specified, then a square is
        constructed basing on the no of cells.
        The above two attributes are suggestions which are respected unless
        they are not feasible. One required condition is that, the number of
        columns is odd when row_groups is greater than 1.
    row_groups: int
        No of groups the rows should be divided into. This can be used to draw
        more square cells for each of the groups

    Layout Attributes

    map_margin: dict (default: {top=50, bottom=50, left=50, right=50})
        Dictionary containing the top, bottom, left and right margins. The user
        is responsible for making sure that the width and height are greater
        than the sum of the margins.
    min_aspect_ratio: float
         minimum width / height ratio of the figure
    max_aspect_ratio: float
         maximum width / height ratio of the figure


    Display Attributes

    colors: list of colors
        Colors for each of the groups which are cycled over to cover all the
        groups
    title: string
        Title of the Market Map
    title_style: dict
        CSS style for the title of the Market Map
    stroke: color
        Stroke of each of the cells of the market map
    group_stroke: color
        Stroke of the border for the group of cells corresponding to a group
    selected_stroke: color
        stroke for the selected cells
    hovered_stroke: color
        stroke for the cell being hovered on
    font_style: dict
        CSS style for the text of each cell

    Other Attributes

    enable_select: bool
        boolean to control the ability to select the cells of the map by
        clicking
    enable_hover: bool
        boolean to control if the map should be aware of which cell is being
        hovered on. If it is set to False, tooltip will not be displayed

    Note
    ----

    The aspect ratios stand for width / height ratios.

     - If the available space is within bounds in terms of min and max aspect
       ratio, we use the entire available space.
     - If the available space is too oblong horizontally, we use the client
       height and the width that corresponds max_aspect_ratio (maximize width
       under the constraints).
     - If the available space is too oblong vertically, we use the client width
       and the height that corresponds to min_aspect_ratio (maximize height
       under the constraint).
       This corresponds to maximizing the area under the constraints.

    Default min and max aspect ratio are both equal to 16 / 9.
    