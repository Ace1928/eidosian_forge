from plotly.basedatatypes import BaseTraceType as _BaseTraceType
import copy as _copy
class Sankey(_BaseTraceType):
    _parent_path_str = ''
    _path_str = 'sankey'
    _valid_props = {'arrangement', 'customdata', 'customdatasrc', 'domain', 'hoverinfo', 'hoverlabel', 'ids', 'idssrc', 'legend', 'legendgrouptitle', 'legendrank', 'legendwidth', 'link', 'meta', 'metasrc', 'name', 'node', 'orientation', 'selectedpoints', 'stream', 'textfont', 'type', 'uid', 'uirevision', 'valueformat', 'valuesuffix', 'visible'}

    @property
    def arrangement(self):
        """
        If value is `snap` (the default), the node arrangement is
        assisted by automatic snapping of elements to preserve space
        between nodes specified via `nodepad`. If value is
        `perpendicular`, the nodes can only move along a line
        perpendicular to the flow. If value is `freeform`, the nodes
        can freely move on the plane. If value is `fixed`, the nodes
        are stationary.

        The 'arrangement' property is an enumeration that may be specified as:
          - One of the following enumeration values:
                ['snap', 'perpendicular', 'freeform', 'fixed']

        Returns
        -------
        Any
        """
        return self['arrangement']

    @arrangement.setter
    def arrangement(self, val):
        self['arrangement'] = val

    @property
    def customdata(self):
        """
        Assigns extra data each datum. This may be useful when
        listening to hover, click and selection events. Note that,
        "scatter" traces also appends customdata items in the markers
        DOM elements

        The 'customdata' property is an array that may be specified as a tuple,
        list, numpy array, or pandas Series

        Returns
        -------
        numpy.ndarray
        """
        return self['customdata']

    @customdata.setter
    def customdata(self, val):
        self['customdata'] = val

    @property
    def customdatasrc(self):
        """
        Sets the source reference on Chart Studio Cloud for
        `customdata`.

        The 'customdatasrc' property must be specified as a string or
        as a plotly.grid_objs.Column object

        Returns
        -------
        str
        """
        return self['customdatasrc']

    @customdatasrc.setter
    def customdatasrc(self, val):
        self['customdatasrc'] = val

    @property
    def domain(self):
        """
        The 'domain' property is an instance of Domain
        that may be specified as:
          - An instance of :class:`plotly.graph_objs.sankey.Domain`
          - A dict of string/value properties that will be passed
            to the Domain constructor

            Supported dict properties:

                column
                    If there is a layout grid, use the domain for
                    this column in the grid for this sankey trace .
                row
                    If there is a layout grid, use the domain for
                    this row in the grid for this sankey trace .
                x
                    Sets the horizontal domain of this sankey trace
                    (in plot fraction).
                y
                    Sets the vertical domain of this sankey trace
                    (in plot fraction).

        Returns
        -------
        plotly.graph_objs.sankey.Domain
        """
        return self['domain']

    @domain.setter
    def domain(self, val):
        self['domain'] = val

    @property
    def hoverinfo(self):
        """
        Determines which trace information appear on hover. If `none`
        or `skip` are set, no information is displayed upon hovering.
        But, if `none` is set, click and hover events are still fired.
        Note that this attribute is superseded by `node.hoverinfo` and
        `node.hoverinfo` for nodes and links respectively.

        The 'hoverinfo' property is a flaglist and may be specified
        as a string containing:
          - Any combination of [] joined with '+' characters
            (e.g. '')
            OR exactly one of ['all', 'none', 'skip'] (e.g. 'skip')

        Returns
        -------
        Any
        """
        return self['hoverinfo']

    @hoverinfo.setter
    def hoverinfo(self, val):
        self['hoverinfo'] = val

    @property
    def hoverlabel(self):
        """
        The 'hoverlabel' property is an instance of Hoverlabel
        that may be specified as:
          - An instance of :class:`plotly.graph_objs.sankey.Hoverlabel`
          - A dict of string/value properties that will be passed
            to the Hoverlabel constructor

            Supported dict properties:

                align
                    Sets the horizontal alignment of the text
                    content within hover label box. Has an effect
                    only if the hover label text spans more two or
                    more lines
                alignsrc
                    Sets the source reference on Chart Studio Cloud
                    for `align`.
                bgcolor
                    Sets the background color of the hover labels
                    for this trace
                bgcolorsrc
                    Sets the source reference on Chart Studio Cloud
                    for `bgcolor`.
                bordercolor
                    Sets the border color of the hover labels for
                    this trace.
                bordercolorsrc
                    Sets the source reference on Chart Studio Cloud
                    for `bordercolor`.
                font
                    Sets the font used in hover labels.
                namelength
                    Sets the default length (in number of
                    characters) of the trace name in the hover
                    labels for all traces. -1 shows the whole name
                    regardless of length. 0-3 shows the first 0-3
                    characters, and an integer >3 will show the
                    whole name if it is less than that many
                    characters, but if it is longer, will truncate
                    to `namelength - 3` characters and add an
                    ellipsis.
                namelengthsrc
                    Sets the source reference on Chart Studio Cloud
                    for `namelength`.

        Returns
        -------
        plotly.graph_objs.sankey.Hoverlabel
        """
        return self['hoverlabel']

    @hoverlabel.setter
    def hoverlabel(self, val):
        self['hoverlabel'] = val

    @property
    def ids(self):
        """
        Assigns id labels to each datum. These ids for object constancy
        of data points during animation. Should be an array of strings,
        not numbers or any other type.

        The 'ids' property is an array that may be specified as a tuple,
        list, numpy array, or pandas Series

        Returns
        -------
        numpy.ndarray
        """
        return self['ids']

    @ids.setter
    def ids(self, val):
        self['ids'] = val

    @property
    def idssrc(self):
        """
        Sets the source reference on Chart Studio Cloud for `ids`.

        The 'idssrc' property must be specified as a string or
        as a plotly.grid_objs.Column object

        Returns
        -------
        str
        """
        return self['idssrc']

    @idssrc.setter
    def idssrc(self, val):
        self['idssrc'] = val

    @property
    def legend(self):
        """
        Sets the reference to a legend to show this trace in.
        References to these legends are "legend", "legend2", "legend3",
        etc. Settings for these legends are set in the layout, under
        `layout.legend`, `layout.legend2`, etc.

        The 'legend' property is an identifier of a particular
        subplot, of type 'legend', that may be specified as the string 'legend'
        optionally followed by an integer >= 1
        (e.g. 'legend', 'legend1', 'legend2', 'legend3', etc.)

        Returns
        -------
        str
        """
        return self['legend']

    @legend.setter
    def legend(self, val):
        self['legend'] = val

    @property
    def legendgrouptitle(self):
        """
        The 'legendgrouptitle' property is an instance of Legendgrouptitle
        that may be specified as:
          - An instance of :class:`plotly.graph_objs.sankey.Legendgrouptitle`
          - A dict of string/value properties that will be passed
            to the Legendgrouptitle constructor

            Supported dict properties:

                font
                    Sets this legend group's title font.
                text
                    Sets the title of the legend group.

        Returns
        -------
        plotly.graph_objs.sankey.Legendgrouptitle
        """
        return self['legendgrouptitle']

    @legendgrouptitle.setter
    def legendgrouptitle(self, val):
        self['legendgrouptitle'] = val

    @property
    def legendrank(self):
        """
        Sets the legend rank for this trace. Items and groups with
        smaller ranks are presented on top/left side while with
        "reversed" `legend.traceorder` they are on bottom/right side.
        The default legendrank is 1000, so that you can use ranks less
        than 1000 to place certain items before all unranked items, and
        ranks greater than 1000 to go after all unranked items. When
        having unranked or equal rank items shapes would be displayed
        after traces i.e. according to their order in data and layout.

        The 'legendrank' property is a number and may be specified as:
          - An int or float

        Returns
        -------
        int|float
        """
        return self['legendrank']

    @legendrank.setter
    def legendrank(self, val):
        self['legendrank'] = val

    @property
    def legendwidth(self):
        """
        Sets the width (in px or fraction) of the legend for this
        trace.

        The 'legendwidth' property is a number and may be specified as:
          - An int or float in the interval [0, inf]

        Returns
        -------
        int|float
        """
        return self['legendwidth']

    @legendwidth.setter
    def legendwidth(self, val):
        self['legendwidth'] = val

    @property
    def link(self):
        """
        The links of the Sankey plot.

        The 'link' property is an instance of Link
        that may be specified as:
          - An instance of :class:`plotly.graph_objs.sankey.Link`
          - A dict of string/value properties that will be passed
            to the Link constructor

            Supported dict properties:

                arrowlen
                    Sets the length (in px) of the links arrow, if
                    0 no arrow will be drawn.
                color
                    Sets the `link` color. It can be a single
                    value, or an array for specifying color for
                    each `link`. If `link.color` is omitted, then
                    by default, a translucent grey link will be
                    used.
                colorscales
                    A tuple of :class:`plotly.graph_objects.sankey.
                    link.Colorscale` instances or dicts with
                    compatible properties
                colorscaledefaults
                    When used in a template (as layout.template.dat
                    a.sankey.link.colorscaledefaults), sets the
                    default property values to use for elements of
                    sankey.link.colorscales
                colorsrc
                    Sets the source reference on Chart Studio Cloud
                    for `color`.
                customdata
                    Assigns extra data to each link.
                customdatasrc
                    Sets the source reference on Chart Studio Cloud
                    for `customdata`.
                hovercolor
                    Sets the `link` hover color. It can be a single
                    value, or an array for specifying hover colors
                    for each `link`. If `link.hovercolor` is
                    omitted, then by default, links will become
                    slightly more opaque when hovered over.
                hovercolorsrc
                    Sets the source reference on Chart Studio Cloud
                    for `hovercolor`.
                hoverinfo
                    Determines which trace information appear when
                    hovering links. If `none` or `skip` are set, no
                    information is displayed upon hovering. But, if
                    `none` is set, click and hover events are still
                    fired.
                hoverlabel
                    :class:`plotly.graph_objects.sankey.link.Hoverl
                    abel` instance or dict with compatible
                    properties
                hovertemplate
                    Template string used for rendering the
                    information that appear on hover box. Note that
                    this will override `hoverinfo`. Variables are
                    inserted using %{variable}, for example "y:
                    %{y}" as well as %{xother}, {%_xother},
                    {%_xother_}, {%xother_}. When showing info for
                    several points, "xother" will be added to those
                    with different x positions from the first
                    point. An underscore before or after
                    "(x|y)other" will add a space on that side,
                    only when this field is shown. Numbers are
                    formatted using d3-format's syntax
                    %{variable:d3-format}, for example "Price:
                    %{y:$.2f}". https://github.com/d3/d3-
                    format/tree/v1.4.5#d3-format for details on the
                    formatting syntax. Dates are formatted using
                    d3-time-format's syntax %{variable|d3-time-
                    format}, for example "Day: %{2019-01-01|%A}".
                    https://github.com/d3/d3-time-
                    format/tree/v2.2.3#locale_format for details on
                    the date formatting syntax. The variables
                    available in `hovertemplate` are the ones
                    emitted as event data described at this link
                    https://plotly.com/javascript/plotlyjs-
                    events/#event-data. Additionally, every
                    attributes that can be specified per-point (the
                    ones that are `arrayOk: true`) are available.
                    Variables `source` and `target` are node
                    objects.Finally, the template string has access
                    to variables `value` and `label`. Anything
                    contained in tag `<extra>` is displayed in the
                    secondary box, for example
                    "<extra>{fullData.name}</extra>". To hide the
                    secondary box completely, use an empty tag
                    `<extra></extra>`.
                hovertemplatesrc
                    Sets the source reference on Chart Studio Cloud
                    for `hovertemplate`.
                label
                    The shown name of the link.
                labelsrc
                    Sets the source reference on Chart Studio Cloud
                    for `label`.
                line
                    :class:`plotly.graph_objects.sankey.link.Line`
                    instance or dict with compatible properties
                source
                    An integer number `[0..nodes.length - 1]` that
                    represents the source node.
                sourcesrc
                    Sets the source reference on Chart Studio Cloud
                    for `source`.
                target
                    An integer number `[0..nodes.length - 1]` that
                    represents the target node.
                targetsrc
                    Sets the source reference on Chart Studio Cloud
                    for `target`.
                value
                    A numeric value representing the flow volume
                    value.
                valuesrc
                    Sets the source reference on Chart Studio Cloud
                    for `value`.

        Returns
        -------
        plotly.graph_objs.sankey.Link
        """
        return self['link']

    @link.setter
    def link(self, val):
        self['link'] = val

    @property
    def meta(self):
        """
        Assigns extra meta information associated with this trace that
        can be used in various text attributes. Attributes such as
        trace `name`, graph, axis and colorbar `title.text`, annotation
        `text` `rangeselector`, `updatemenues` and `sliders` `label`
        text all support `meta`. To access the trace `meta` values in
        an attribute in the same trace, simply use `%{meta[i]}` where
        `i` is the index or key of the `meta` item in question. To
        access trace `meta` in layout attributes, use
        `%{data[n[.meta[i]}` where `i` is the index or key of the
        `meta` and `n` is the trace index.

        The 'meta' property accepts values of any type

        Returns
        -------
        Any|numpy.ndarray
        """
        return self['meta']

    @meta.setter
    def meta(self, val):
        self['meta'] = val

    @property
    def metasrc(self):
        """
        Sets the source reference on Chart Studio Cloud for `meta`.

        The 'metasrc' property must be specified as a string or
        as a plotly.grid_objs.Column object

        Returns
        -------
        str
        """
        return self['metasrc']

    @metasrc.setter
    def metasrc(self, val):
        self['metasrc'] = val

    @property
    def name(self):
        """
        Sets the trace name. The trace name appears as the legend item
        and on hover.

        The 'name' property is a string and must be specified as:
          - A string
          - A number that will be converted to a string

        Returns
        -------
        str
        """
        return self['name']

    @name.setter
    def name(self, val):
        self['name'] = val

    @property
    def node(self):
        """
        The nodes of the Sankey plot.

        The 'node' property is an instance of Node
        that may be specified as:
          - An instance of :class:`plotly.graph_objs.sankey.Node`
          - A dict of string/value properties that will be passed
            to the Node constructor

            Supported dict properties:

                align
                    Sets the alignment method used to position the
                    nodes along the horizontal axis.
                color
                    Sets the `node` color. It can be a single
                    value, or an array for specifying color for
                    each `node`. If `node.color` is omitted, then
                    the default `Plotly` color palette will be
                    cycled through to have a variety of colors.
                    These defaults are not fully opaque, to allow
                    some visibility of what is beneath the node.
                colorsrc
                    Sets the source reference on Chart Studio Cloud
                    for `color`.
                customdata
                    Assigns extra data to each node.
                customdatasrc
                    Sets the source reference on Chart Studio Cloud
                    for `customdata`.
                groups
                    Groups of nodes. Each group is defined by an
                    array with the indices of the nodes it
                    contains. Multiple groups can be specified.
                hoverinfo
                    Determines which trace information appear when
                    hovering nodes. If `none` or `skip` are set, no
                    information is displayed upon hovering. But, if
                    `none` is set, click and hover events are still
                    fired.
                hoverlabel
                    :class:`plotly.graph_objects.sankey.node.Hoverl
                    abel` instance or dict with compatible
                    properties
                hovertemplate
                    Template string used for rendering the
                    information that appear on hover box. Note that
                    this will override `hoverinfo`. Variables are
                    inserted using %{variable}, for example "y:
                    %{y}" as well as %{xother}, {%_xother},
                    {%_xother_}, {%xother_}. When showing info for
                    several points, "xother" will be added to those
                    with different x positions from the first
                    point. An underscore before or after
                    "(x|y)other" will add a space on that side,
                    only when this field is shown. Numbers are
                    formatted using d3-format's syntax
                    %{variable:d3-format}, for example "Price:
                    %{y:$.2f}". https://github.com/d3/d3-
                    format/tree/v1.4.5#d3-format for details on the
                    formatting syntax. Dates are formatted using
                    d3-time-format's syntax %{variable|d3-time-
                    format}, for example "Day: %{2019-01-01|%A}".
                    https://github.com/d3/d3-time-
                    format/tree/v2.2.3#locale_format for details on
                    the date formatting syntax. The variables
                    available in `hovertemplate` are the ones
                    emitted as event data described at this link
                    https://plotly.com/javascript/plotlyjs-
                    events/#event-data. Additionally, every
                    attributes that can be specified per-point (the
                    ones that are `arrayOk: true`) are available.
                    Variables `sourceLinks` and `targetLinks` are
                    arrays of link objects.Finally, the template
                    string has access to variables `value` and
                    `label`. Anything contained in tag `<extra>` is
                    displayed in the secondary box, for example
                    "<extra>{fullData.name}</extra>". To hide the
                    secondary box completely, use an empty tag
                    `<extra></extra>`.
                hovertemplatesrc
                    Sets the source reference on Chart Studio Cloud
                    for `hovertemplate`.
                label
                    The shown name of the node.
                labelsrc
                    Sets the source reference on Chart Studio Cloud
                    for `label`.
                line
                    :class:`plotly.graph_objects.sankey.node.Line`
                    instance or dict with compatible properties
                pad
                    Sets the padding (in px) between the `nodes`.
                thickness
                    Sets the thickness (in px) of the `nodes`.
                x
                    The normalized horizontal position of the node.
                xsrc
                    Sets the source reference on Chart Studio Cloud
                    for `x`.
                y
                    The normalized vertical position of the node.
                ysrc
                    Sets the source reference on Chart Studio Cloud
                    for `y`.

        Returns
        -------
        plotly.graph_objs.sankey.Node
        """
        return self['node']

    @node.setter
    def node(self, val):
        self['node'] = val

    @property
    def orientation(self):
        """
        Sets the orientation of the Sankey diagram.

        The 'orientation' property is an enumeration that may be specified as:
          - One of the following enumeration values:
                ['v', 'h']

        Returns
        -------
        Any
        """
        return self['orientation']

    @orientation.setter
    def orientation(self, val):
        self['orientation'] = val

    @property
    def selectedpoints(self):
        """
        Array containing integer indices of selected points. Has an
        effect only for traces that support selections. Note that an
        empty array means an empty selection where the `unselected` are
        turned on for all points, whereas, any other non-array values
        means no selection all where the `selected` and `unselected`
        styles have no effect.

        The 'selectedpoints' property accepts values of any type

        Returns
        -------
        Any
        """
        return self['selectedpoints']

    @selectedpoints.setter
    def selectedpoints(self, val):
        self['selectedpoints'] = val

    @property
    def stream(self):
        """
        The 'stream' property is an instance of Stream
        that may be specified as:
          - An instance of :class:`plotly.graph_objs.sankey.Stream`
          - A dict of string/value properties that will be passed
            to the Stream constructor

            Supported dict properties:

                maxpoints
                    Sets the maximum number of points to keep on
                    the plots from an incoming stream. If
                    `maxpoints` is set to 50, only the newest 50
                    points will be displayed on the plot.
                token
                    The stream id number links a data trace on a
                    plot with a stream. See https://chart-
                    studio.plotly.com/settings for more details.

        Returns
        -------
        plotly.graph_objs.sankey.Stream
        """
        return self['stream']

    @stream.setter
    def stream(self, val):
        self['stream'] = val

    @property
    def textfont(self):
        """
        Sets the font for node labels

        The 'textfont' property is an instance of Textfont
        that may be specified as:
          - An instance of :class:`plotly.graph_objs.sankey.Textfont`
          - A dict of string/value properties that will be passed
            to the Textfont constructor

            Supported dict properties:

                color

                family
                    HTML font family - the typeface that will be
                    applied by the web browser. The web browser
                    will only be able to apply a font if it is
                    available on the system which it operates.
                    Provide multiple font families, separated by
                    commas, to indicate the preference in which to
                    apply fonts if they aren't available on the
                    system. The Chart Studio Cloud (at
                    https://chart-studio.plotly.com or on-premise)
                    generates images on a server, where only a
                    select number of fonts are installed and
                    supported. These include "Arial", "Balto",
                    "Courier New", "Droid Sans",, "Droid Serif",
                    "Droid Sans Mono", "Gravitas One", "Old
                    Standard TT", "Open Sans", "Overpass", "PT Sans
                    Narrow", "Raleway", "Times New Roman".
                size

        Returns
        -------
        plotly.graph_objs.sankey.Textfont
        """
        return self['textfont']

    @textfont.setter
    def textfont(self, val):
        self['textfont'] = val

    @property
    def uid(self):
        """
        Assign an id to this trace, Use this to provide object
        constancy between traces during animations and transitions.

        The 'uid' property is a string and must be specified as:
          - A string
          - A number that will be converted to a string

        Returns
        -------
        str
        """
        return self['uid']

    @uid.setter
    def uid(self, val):
        self['uid'] = val

    @property
    def uirevision(self):
        """
        Controls persistence of some user-driven changes to the trace:
        `constraintrange` in `parcoords` traces, as well as some
        `editable: true` modifications such as `name` and
        `colorbar.title`. Defaults to `layout.uirevision`. Note that
        other user-driven trace attribute changes are controlled by
        `layout` attributes: `trace.visible` is controlled by
        `layout.legend.uirevision`, `selectedpoints` is controlled by
        `layout.selectionrevision`, and `colorbar.(x|y)` (accessible
        with `config: {editable: true}`) is controlled by
        `layout.editrevision`. Trace changes are tracked by `uid`,
        which only falls back on trace index if no `uid` is provided.
        So if your app can add/remove traces before the end of the
        `data` array, such that the same trace has a different index,
        you can still preserve user-driven changes if you give each
        trace a `uid` that stays with it as it moves.

        The 'uirevision' property accepts values of any type

        Returns
        -------
        Any
        """
        return self['uirevision']

    @uirevision.setter
    def uirevision(self, val):
        self['uirevision'] = val

    @property
    def valueformat(self):
        """
        Sets the value formatting rule using d3 formatting mini-
        languages which are very similar to those in Python. For
        numbers, see:
        https://github.com/d3/d3-format/tree/v1.4.5#d3-format.

        The 'valueformat' property is a string and must be specified as:
          - A string
          - A number that will be converted to a string

        Returns
        -------
        str
        """
        return self['valueformat']

    @valueformat.setter
    def valueformat(self, val):
        self['valueformat'] = val

    @property
    def valuesuffix(self):
        """
        Adds a unit to follow the value in the hover tooltip. Add a
        space if a separation is necessary from the value.

        The 'valuesuffix' property is a string and must be specified as:
          - A string
          - A number that will be converted to a string

        Returns
        -------
        str
        """
        return self['valuesuffix']

    @valuesuffix.setter
    def valuesuffix(self, val):
        self['valuesuffix'] = val

    @property
    def visible(self):
        """
        Determines whether or not this trace is visible. If
        "legendonly", the trace is not drawn, but can appear as a
        legend item (provided that the legend itself is visible).

        The 'visible' property is an enumeration that may be specified as:
          - One of the following enumeration values:
                [True, False, 'legendonly']

        Returns
        -------
        Any
        """
        return self['visible']

    @visible.setter
    def visible(self, val):
        self['visible'] = val

    @property
    def type(self):
        return self._props['type']

    @property
    def _prop_descriptions(self):
        return '        arrangement\n            If value is `snap` (the default), the node arrangement\n            is assisted by automatic snapping of elements to\n            preserve space between nodes specified via `nodepad`.\n            If value is `perpendicular`, the nodes can only move\n            along a line perpendicular to the flow. If value is\n            `freeform`, the nodes can freely move on the plane. If\n            value is `fixed`, the nodes are stationary.\n        customdata\n            Assigns extra data each datum. This may be useful when\n            listening to hover, click and selection events. Note\n            that, "scatter" traces also appends customdata items in\n            the markers DOM elements\n        customdatasrc\n            Sets the source reference on Chart Studio Cloud for\n            `customdata`.\n        domain\n            :class:`plotly.graph_objects.sankey.Domain` instance or\n            dict with compatible properties\n        hoverinfo\n            Determines which trace information appear on hover. If\n            `none` or `skip` are set, no information is displayed\n            upon hovering. But, if `none` is set, click and hover\n            events are still fired. Note that this attribute is\n            superseded by `node.hoverinfo` and `node.hoverinfo` for\n            nodes and links respectively.\n        hoverlabel\n            :class:`plotly.graph_objects.sankey.Hoverlabel`\n            instance or dict with compatible properties\n        ids\n            Assigns id labels to each datum. These ids for object\n            constancy of data points during animation. Should be an\n            array of strings, not numbers or any other type.\n        idssrc\n            Sets the source reference on Chart Studio Cloud for\n            `ids`.\n        legend\n            Sets the reference to a legend to show this trace in.\n            References to these legends are "legend", "legend2",\n            "legend3", etc. Settings for these legends are set in\n            the layout, under `layout.legend`, `layout.legend2`,\n            etc.\n        legendgrouptitle\n            :class:`plotly.graph_objects.sankey.Legendgrouptitle`\n            instance or dict with compatible properties\n        legendrank\n            Sets the legend rank for this trace. Items and groups\n            with smaller ranks are presented on top/left side while\n            with "reversed" `legend.traceorder` they are on\n            bottom/right side. The default legendrank is 1000, so\n            that you can use ranks less than 1000 to place certain\n            items before all unranked items, and ranks greater than\n            1000 to go after all unranked items. When having\n            unranked or equal rank items shapes would be displayed\n            after traces i.e. according to their order in data and\n            layout.\n        legendwidth\n            Sets the width (in px or fraction) of the legend for\n            this trace.\n        link\n            The links of the Sankey plot.\n        meta\n            Assigns extra meta information associated with this\n            trace that can be used in various text attributes.\n            Attributes such as trace `name`, graph, axis and\n            colorbar `title.text`, annotation `text`\n            `rangeselector`, `updatemenues` and `sliders` `label`\n            text all support `meta`. To access the trace `meta`\n            values in an attribute in the same trace, simply use\n            `%{meta[i]}` where `i` is the index or key of the\n            `meta` item in question. To access trace `meta` in\n            layout attributes, use `%{data[n[.meta[i]}` where `i`\n            is the index or key of the `meta` and `n` is the trace\n            index.\n        metasrc\n            Sets the source reference on Chart Studio Cloud for\n            `meta`.\n        name\n            Sets the trace name. The trace name appears as the\n            legend item and on hover.\n        node\n            The nodes of the Sankey plot.\n        orientation\n            Sets the orientation of the Sankey diagram.\n        selectedpoints\n            Array containing integer indices of selected points.\n            Has an effect only for traces that support selections.\n            Note that an empty array means an empty selection where\n            the `unselected` are turned on for all points, whereas,\n            any other non-array values means no selection all where\n            the `selected` and `unselected` styles have no effect.\n        stream\n            :class:`plotly.graph_objects.sankey.Stream` instance or\n            dict with compatible properties\n        textfont\n            Sets the font for node labels\n        uid\n            Assign an id to this trace, Use this to provide object\n            constancy between traces during animations and\n            transitions.\n        uirevision\n            Controls persistence of some user-driven changes to the\n            trace: `constraintrange` in `parcoords` traces, as well\n            as some `editable: true` modifications such as `name`\n            and `colorbar.title`. Defaults to `layout.uirevision`.\n            Note that other user-driven trace attribute changes are\n            controlled by `layout` attributes: `trace.visible` is\n            controlled by `layout.legend.uirevision`,\n            `selectedpoints` is controlled by\n            `layout.selectionrevision`, and `colorbar.(x|y)`\n            (accessible with `config: {editable: true}`) is\n            controlled by `layout.editrevision`. Trace changes are\n            tracked by `uid`, which only falls back on trace index\n            if no `uid` is provided. So if your app can add/remove\n            traces before the end of the `data` array, such that\n            the same trace has a different index, you can still\n            preserve user-driven changes if you give each trace a\n            `uid` that stays with it as it moves.\n        valueformat\n            Sets the value formatting rule using d3 formatting\n            mini-languages which are very similar to those in\n            Python. For numbers, see:\n            https://github.com/d3/d3-format/tree/v1.4.5#d3-format.\n        valuesuffix\n            Adds a unit to follow the value in the hover tooltip.\n            Add a space if a separation is necessary from the\n            value.\n        visible\n            Determines whether or not this trace is visible. If\n            "legendonly", the trace is not drawn, but can appear as\n            a legend item (provided that the legend itself is\n            visible).\n        '

    def __init__(self, arg=None, arrangement=None, customdata=None, customdatasrc=None, domain=None, hoverinfo=None, hoverlabel=None, ids=None, idssrc=None, legend=None, legendgrouptitle=None, legendrank=None, legendwidth=None, link=None, meta=None, metasrc=None, name=None, node=None, orientation=None, selectedpoints=None, stream=None, textfont=None, uid=None, uirevision=None, valueformat=None, valuesuffix=None, visible=None, **kwargs):
        """
        Construct a new Sankey object

        Sankey plots for network flow data analysis. The nodes are
        specified in `nodes` and the links between sources and targets
        in `links`. The colors are set in `nodes[i].color` and
        `links[i].color`, otherwise defaults are used.

        Parameters
        ----------
        arg
            dict of properties compatible with this constructor or
            an instance of :class:`plotly.graph_objs.Sankey`
        arrangement
            If value is `snap` (the default), the node arrangement
            is assisted by automatic snapping of elements to
            preserve space between nodes specified via `nodepad`.
            If value is `perpendicular`, the nodes can only move
            along a line perpendicular to the flow. If value is
            `freeform`, the nodes can freely move on the plane. If
            value is `fixed`, the nodes are stationary.
        customdata
            Assigns extra data each datum. This may be useful when
            listening to hover, click and selection events. Note
            that, "scatter" traces also appends customdata items in
            the markers DOM elements
        customdatasrc
            Sets the source reference on Chart Studio Cloud for
            `customdata`.
        domain
            :class:`plotly.graph_objects.sankey.Domain` instance or
            dict with compatible properties
        hoverinfo
            Determines which trace information appear on hover. If
            `none` or `skip` are set, no information is displayed
            upon hovering. But, if `none` is set, click and hover
            events are still fired. Note that this attribute is
            superseded by `node.hoverinfo` and `node.hoverinfo` for
            nodes and links respectively.
        hoverlabel
            :class:`plotly.graph_objects.sankey.Hoverlabel`
            instance or dict with compatible properties
        ids
            Assigns id labels to each datum. These ids for object
            constancy of data points during animation. Should be an
            array of strings, not numbers or any other type.
        idssrc
            Sets the source reference on Chart Studio Cloud for
            `ids`.
        legend
            Sets the reference to a legend to show this trace in.
            References to these legends are "legend", "legend2",
            "legend3", etc. Settings for these legends are set in
            the layout, under `layout.legend`, `layout.legend2`,
            etc.
        legendgrouptitle
            :class:`plotly.graph_objects.sankey.Legendgrouptitle`
            instance or dict with compatible properties
        legendrank
            Sets the legend rank for this trace. Items and groups
            with smaller ranks are presented on top/left side while
            with "reversed" `legend.traceorder` they are on
            bottom/right side. The default legendrank is 1000, so
            that you can use ranks less than 1000 to place certain
            items before all unranked items, and ranks greater than
            1000 to go after all unranked items. When having
            unranked or equal rank items shapes would be displayed
            after traces i.e. according to their order in data and
            layout.
        legendwidth
            Sets the width (in px or fraction) of the legend for
            this trace.
        link
            The links of the Sankey plot.
        meta
            Assigns extra meta information associated with this
            trace that can be used in various text attributes.
            Attributes such as trace `name`, graph, axis and
            colorbar `title.text`, annotation `text`
            `rangeselector`, `updatemenues` and `sliders` `label`
            text all support `meta`. To access the trace `meta`
            values in an attribute in the same trace, simply use
            `%{meta[i]}` where `i` is the index or key of the
            `meta` item in question. To access trace `meta` in
            layout attributes, use `%{data[n[.meta[i]}` where `i`
            is the index or key of the `meta` and `n` is the trace
            index.
        metasrc
            Sets the source reference on Chart Studio Cloud for
            `meta`.
        name
            Sets the trace name. The trace name appears as the
            legend item and on hover.
        node
            The nodes of the Sankey plot.
        orientation
            Sets the orientation of the Sankey diagram.
        selectedpoints
            Array containing integer indices of selected points.
            Has an effect only for traces that support selections.
            Note that an empty array means an empty selection where
            the `unselected` are turned on for all points, whereas,
            any other non-array values means no selection all where
            the `selected` and `unselected` styles have no effect.
        stream
            :class:`plotly.graph_objects.sankey.Stream` instance or
            dict with compatible properties
        textfont
            Sets the font for node labels
        uid
            Assign an id to this trace, Use this to provide object
            constancy between traces during animations and
            transitions.
        uirevision
            Controls persistence of some user-driven changes to the
            trace: `constraintrange` in `parcoords` traces, as well
            as some `editable: true` modifications such as `name`
            and `colorbar.title`. Defaults to `layout.uirevision`.
            Note that other user-driven trace attribute changes are
            controlled by `layout` attributes: `trace.visible` is
            controlled by `layout.legend.uirevision`,
            `selectedpoints` is controlled by
            `layout.selectionrevision`, and `colorbar.(x|y)`
            (accessible with `config: {editable: true}`) is
            controlled by `layout.editrevision`. Trace changes are
            tracked by `uid`, which only falls back on trace index
            if no `uid` is provided. So if your app can add/remove
            traces before the end of the `data` array, such that
            the same trace has a different index, you can still
            preserve user-driven changes if you give each trace a
            `uid` that stays with it as it moves.
        valueformat
            Sets the value formatting rule using d3 formatting
            mini-languages which are very similar to those in
            Python. For numbers, see:
            https://github.com/d3/d3-format/tree/v1.4.5#d3-format.
        valuesuffix
            Adds a unit to follow the value in the hover tooltip.
            Add a space if a separation is necessary from the
            value.
        visible
            Determines whether or not this trace is visible. If
            "legendonly", the trace is not drawn, but can appear as
            a legend item (provided that the legend itself is
            visible).

        Returns
        -------
        Sankey
        """
        super(Sankey, self).__init__('sankey')
        if '_parent' in kwargs:
            self._parent = kwargs['_parent']
            return
        if arg is None:
            arg = {}
        elif isinstance(arg, self.__class__):
            arg = arg.to_plotly_json()
        elif isinstance(arg, dict):
            arg = _copy.copy(arg)
        else:
            raise ValueError('The first argument to the plotly.graph_objs.Sankey\nconstructor must be a dict or\nan instance of :class:`plotly.graph_objs.Sankey`')
        self._skip_invalid = kwargs.pop('skip_invalid', False)
        self._validate = kwargs.pop('_validate', True)
        _v = arg.pop('arrangement', None)
        _v = arrangement if arrangement is not None else _v
        if _v is not None:
            self['arrangement'] = _v
        _v = arg.pop('customdata', None)
        _v = customdata if customdata is not None else _v
        if _v is not None:
            self['customdata'] = _v
        _v = arg.pop('customdatasrc', None)
        _v = customdatasrc if customdatasrc is not None else _v
        if _v is not None:
            self['customdatasrc'] = _v
        _v = arg.pop('domain', None)
        _v = domain if domain is not None else _v
        if _v is not None:
            self['domain'] = _v
        _v = arg.pop('hoverinfo', None)
        _v = hoverinfo if hoverinfo is not None else _v
        if _v is not None:
            self['hoverinfo'] = _v
        _v = arg.pop('hoverlabel', None)
        _v = hoverlabel if hoverlabel is not None else _v
        if _v is not None:
            self['hoverlabel'] = _v
        _v = arg.pop('ids', None)
        _v = ids if ids is not None else _v
        if _v is not None:
            self['ids'] = _v
        _v = arg.pop('idssrc', None)
        _v = idssrc if idssrc is not None else _v
        if _v is not None:
            self['idssrc'] = _v
        _v = arg.pop('legend', None)
        _v = legend if legend is not None else _v
        if _v is not None:
            self['legend'] = _v
        _v = arg.pop('legendgrouptitle', None)
        _v = legendgrouptitle if legendgrouptitle is not None else _v
        if _v is not None:
            self['legendgrouptitle'] = _v
        _v = arg.pop('legendrank', None)
        _v = legendrank if legendrank is not None else _v
        if _v is not None:
            self['legendrank'] = _v
        _v = arg.pop('legendwidth', None)
        _v = legendwidth if legendwidth is not None else _v
        if _v is not None:
            self['legendwidth'] = _v
        _v = arg.pop('link', None)
        _v = link if link is not None else _v
        if _v is not None:
            self['link'] = _v
        _v = arg.pop('meta', None)
        _v = meta if meta is not None else _v
        if _v is not None:
            self['meta'] = _v
        _v = arg.pop('metasrc', None)
        _v = metasrc if metasrc is not None else _v
        if _v is not None:
            self['metasrc'] = _v
        _v = arg.pop('name', None)
        _v = name if name is not None else _v
        if _v is not None:
            self['name'] = _v
        _v = arg.pop('node', None)
        _v = node if node is not None else _v
        if _v is not None:
            self['node'] = _v
        _v = arg.pop('orientation', None)
        _v = orientation if orientation is not None else _v
        if _v is not None:
            self['orientation'] = _v
        _v = arg.pop('selectedpoints', None)
        _v = selectedpoints if selectedpoints is not None else _v
        if _v is not None:
            self['selectedpoints'] = _v
        _v = arg.pop('stream', None)
        _v = stream if stream is not None else _v
        if _v is not None:
            self['stream'] = _v
        _v = arg.pop('textfont', None)
        _v = textfont if textfont is not None else _v
        if _v is not None:
            self['textfont'] = _v
        _v = arg.pop('uid', None)
        _v = uid if uid is not None else _v
        if _v is not None:
            self['uid'] = _v
        _v = arg.pop('uirevision', None)
        _v = uirevision if uirevision is not None else _v
        if _v is not None:
            self['uirevision'] = _v
        _v = arg.pop('valueformat', None)
        _v = valueformat if valueformat is not None else _v
        if _v is not None:
            self['valueformat'] = _v
        _v = arg.pop('valuesuffix', None)
        _v = valuesuffix if valuesuffix is not None else _v
        if _v is not None:
            self['valuesuffix'] = _v
        _v = arg.pop('visible', None)
        _v = visible if visible is not None else _v
        if _v is not None:
            self['visible'] = _v
        self._props['type'] = 'sankey'
        arg.pop('type', None)
        self._process_kwargs(**dict(arg, **kwargs))
        self._skip_invalid = False