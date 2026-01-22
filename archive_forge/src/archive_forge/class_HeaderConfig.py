from typing import Any, Literal, Union, Protocol, Sequence, List
from typing import Dict as TypingDict
from typing import Generator as TypingGenerator
from altair.utils.schemapi import SchemaBase, Undefined, UndefinedType, _subclasses
import pkgutil
import json
class HeaderConfig(VegaLiteSchema):
    """HeaderConfig schema wrapper

    Parameters
    ----------

    format : str, dict, :class:`Dict`
        When used with the default ``"number"`` and ``"time"`` format type, the text
        formatting pattern for labels of guides (axes, legends, headers) and text marks.


        * If the format type is ``"number"`` (e.g., for quantitative fields), this is D3's
          `number format pattern <https://github.com/d3/d3-format#locale_format>`__.
        * If the format type is ``"time"`` (e.g., for temporal fields), this is D3's `time
          format pattern <https://github.com/d3/d3-time-format#locale_format>`__.

        See the `format documentation <https://vega.github.io/vega-lite/docs/format.html>`__
        for more examples.

        When used with a `custom formatType
        <https://vega.github.io/vega-lite/docs/config.html#custom-format-type>`__, this
        value will be passed as ``format`` alongside ``datum.value`` to the registered
        function.

        **Default value:**  Derived from `numberFormat
        <https://vega.github.io/vega-lite/docs/config.html#format>`__ config for number
        format and from `timeFormat
        <https://vega.github.io/vega-lite/docs/config.html#format>`__ config for time
        format.
    formatType : str
        The format type for labels. One of ``"number"``, ``"time"``, or a `registered custom
        format type
        <https://vega.github.io/vega-lite/docs/config.html#custom-format-type>`__.

        **Default value:**


        * ``"time"`` for temporal fields and ordinal and nominal fields with ``timeUnit``.
        * ``"number"`` for quantitative fields as well as ordinal and nominal fields without
          ``timeUnit``.
    labelAlign : dict, :class:`Align`, :class:`ExprRef`, Literal['left', 'center', 'right']
        Horizontal text alignment of header labels. One of ``"left"``, ``"center"``, or
        ``"right"``.
    labelAnchor : :class:`TitleAnchor`, Literal[None, 'start', 'middle', 'end']
        The anchor position for placing the labels. One of ``"start"``, ``"middle"``, or
        ``"end"``. For example, with a label orientation of top these anchor positions map
        to a left-, center-, or right-aligned label.
    labelAngle : float
        The rotation angle of the header labels.

        **Default value:** ``0`` for column header, ``-90`` for row header.
    labelBaseline : str, dict, :class:`ExprRef`, :class:`Baseline`, :class:`TextBaseline`, Literal['top', 'middle', 'bottom']
        The vertical text baseline for the header labels. One of ``"alphabetic"`` (default),
        ``"top"``, ``"middle"``, ``"bottom"``, ``"line-top"``, or ``"line-bottom"``. The
        ``"line-top"`` and ``"line-bottom"`` values operate similarly to ``"top"`` and
        ``"bottom"``, but are calculated relative to the ``titleLineHeight`` rather than
        ``titleFontSize`` alone.
    labelColor : str, dict, :class:`Color`, :class:`ExprRef`, :class:`HexColor`, :class:`ColorName`, Literal['black', 'silver', 'gray', 'white', 'maroon', 'red', 'purple', 'fuchsia', 'green', 'lime', 'olive', 'yellow', 'navy', 'blue', 'teal', 'aqua', 'orange', 'aliceblue', 'antiquewhite', 'aquamarine', 'azure', 'beige', 'bisque', 'blanchedalmond', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow', 'limegreen', 'linen', 'magenta', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'oldlace', 'olivedrab', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'whitesmoke', 'yellowgreen', 'rebeccapurple']
        The color of the header label, can be in hex color code or regular color name.
    labelExpr : str
        `Vega expression <https://vega.github.io/vega/docs/expressions/>`__ for customizing
        labels.

        **Note:** The label text and value can be assessed via the ``label`` and ``value``
        properties of the header's backing ``datum`` object.
    labelFont : str, dict, :class:`ExprRef`
        The font of the header label.
    labelFontSize : dict, float, :class:`ExprRef`
        The font size of the header label, in pixels.
    labelFontStyle : str, dict, :class:`ExprRef`, :class:`FontStyle`
        The font style of the header label.
    labelFontWeight : dict, :class:`ExprRef`, :class:`FontWeight`, Literal['normal', 'bold', 'lighter', 'bolder', 100, 200, 300, 400, 500, 600, 700, 800, 900]
        The font weight of the header label.
    labelLimit : dict, float, :class:`ExprRef`
        The maximum length of the header label in pixels. The text value will be
        automatically truncated if the rendered size exceeds the limit.

        **Default value:** ``0``, indicating no limit
    labelLineHeight : dict, float, :class:`ExprRef`
        Line height in pixels for multi-line header labels or title text with ``"line-top"``
        or ``"line-bottom"`` baseline.
    labelOrient : :class:`Orient`, Literal['left', 'right', 'top', 'bottom']
        The orientation of the header label. One of ``"top"``, ``"bottom"``, ``"left"`` or
        ``"right"``.
    labelPadding : dict, float, :class:`ExprRef`
        The padding, in pixel, between facet header's label and the plot.

        **Default value:** ``10``
    labels : bool
        A boolean flag indicating if labels should be included as part of the header.

        **Default value:** ``true``.
    orient : :class:`Orient`, Literal['left', 'right', 'top', 'bottom']
        Shortcut for setting both labelOrient and titleOrient.
    title : None
        Set to null to disable title for the axis, legend, or header.
    titleAlign : dict, :class:`Align`, :class:`ExprRef`, Literal['left', 'center', 'right']
        Horizontal text alignment (to the anchor) of header titles.
    titleAnchor : :class:`TitleAnchor`, Literal[None, 'start', 'middle', 'end']
        The anchor position for placing the title. One of ``"start"``, ``"middle"``, or
        ``"end"``. For example, with an orientation of top these anchor positions map to a
        left-, center-, or right-aligned title.
    titleAngle : float
        The rotation angle of the header title.

        **Default value:** ``0``.
    titleBaseline : str, dict, :class:`ExprRef`, :class:`Baseline`, :class:`TextBaseline`, Literal['top', 'middle', 'bottom']
        The vertical text baseline for the header title. One of ``"alphabetic"`` (default),
        ``"top"``, ``"middle"``, ``"bottom"``, ``"line-top"``, or ``"line-bottom"``. The
        ``"line-top"`` and ``"line-bottom"`` values operate similarly to ``"top"`` and
        ``"bottom"``, but are calculated relative to the ``titleLineHeight`` rather than
        ``titleFontSize`` alone.

        **Default value:** ``"middle"``
    titleColor : str, dict, :class:`Color`, :class:`ExprRef`, :class:`HexColor`, :class:`ColorName`, Literal['black', 'silver', 'gray', 'white', 'maroon', 'red', 'purple', 'fuchsia', 'green', 'lime', 'olive', 'yellow', 'navy', 'blue', 'teal', 'aqua', 'orange', 'aliceblue', 'antiquewhite', 'aquamarine', 'azure', 'beige', 'bisque', 'blanchedalmond', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow', 'limegreen', 'linen', 'magenta', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'oldlace', 'olivedrab', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'whitesmoke', 'yellowgreen', 'rebeccapurple']
        Color of the header title, can be in hex color code or regular color name.
    titleFont : str, dict, :class:`ExprRef`
        Font of the header title. (e.g., ``"Helvetica Neue"`` ).
    titleFontSize : dict, float, :class:`ExprRef`
        Font size of the header title.
    titleFontStyle : str, dict, :class:`ExprRef`, :class:`FontStyle`
        The font style of the header title.
    titleFontWeight : dict, :class:`ExprRef`, :class:`FontWeight`, Literal['normal', 'bold', 'lighter', 'bolder', 100, 200, 300, 400, 500, 600, 700, 800, 900]
        Font weight of the header title. This can be either a string (e.g ``"bold"``,
        ``"normal"`` ) or a number ( ``100``, ``200``, ``300``, ..., ``900`` where
        ``"normal"`` = ``400`` and ``"bold"`` = ``700`` ).
    titleLimit : dict, float, :class:`ExprRef`
        The maximum length of the header title in pixels. The text value will be
        automatically truncated if the rendered size exceeds the limit.

        **Default value:** ``0``, indicating no limit
    titleLineHeight : dict, float, :class:`ExprRef`
        Line height in pixels for multi-line header title text or title text with
        ``"line-top"`` or ``"line-bottom"`` baseline.
    titleOrient : :class:`Orient`, Literal['left', 'right', 'top', 'bottom']
        The orientation of the header title. One of ``"top"``, ``"bottom"``, ``"left"`` or
        ``"right"``.
    titlePadding : dict, float, :class:`ExprRef`
        The padding, in pixel, between facet header's title and the label.

        **Default value:** ``10``
    """
    _schema = {'$ref': '#/definitions/HeaderConfig'}

    def __init__(self, format: Union[str, dict, 'SchemaBase', UndefinedType]=Undefined, formatType: Union[str, UndefinedType]=Undefined, labelAlign: Union[dict, '_Parameter', 'SchemaBase', Literal['left', 'center', 'right'], UndefinedType]=Undefined, labelAnchor: Union['SchemaBase', Literal[None, 'start', 'middle', 'end'], UndefinedType]=Undefined, labelAngle: Union[float, UndefinedType]=Undefined, labelBaseline: Union[str, dict, '_Parameter', 'SchemaBase', Literal['top', 'middle', 'bottom'], UndefinedType]=Undefined, labelColor: Union[str, dict, '_Parameter', 'SchemaBase', Literal['black', 'silver', 'gray', 'white', 'maroon', 'red', 'purple', 'fuchsia', 'green', 'lime', 'olive', 'yellow', 'navy', 'blue', 'teal', 'aqua', 'orange', 'aliceblue', 'antiquewhite', 'aquamarine', 'azure', 'beige', 'bisque', 'blanchedalmond', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow', 'limegreen', 'linen', 'magenta', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'oldlace', 'olivedrab', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'whitesmoke', 'yellowgreen', 'rebeccapurple'], UndefinedType]=Undefined, labelExpr: Union[str, UndefinedType]=Undefined, labelFont: Union[str, dict, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, labelFontSize: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, labelFontStyle: Union[str, dict, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, labelFontWeight: Union[dict, '_Parameter', 'SchemaBase', Literal['normal', 'bold', 'lighter', 'bolder', 100, 200, 300, 400, 500, 600, 700, 800, 900], UndefinedType]=Undefined, labelLimit: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, labelLineHeight: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, labelOrient: Union['SchemaBase', Literal['left', 'right', 'top', 'bottom'], UndefinedType]=Undefined, labelPadding: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, labels: Union[bool, UndefinedType]=Undefined, orient: Union['SchemaBase', Literal['left', 'right', 'top', 'bottom'], UndefinedType]=Undefined, title: Union[None, UndefinedType]=Undefined, titleAlign: Union[dict, '_Parameter', 'SchemaBase', Literal['left', 'center', 'right'], UndefinedType]=Undefined, titleAnchor: Union['SchemaBase', Literal[None, 'start', 'middle', 'end'], UndefinedType]=Undefined, titleAngle: Union[float, UndefinedType]=Undefined, titleBaseline: Union[str, dict, '_Parameter', 'SchemaBase', Literal['top', 'middle', 'bottom'], UndefinedType]=Undefined, titleColor: Union[str, dict, '_Parameter', 'SchemaBase', Literal['black', 'silver', 'gray', 'white', 'maroon', 'red', 'purple', 'fuchsia', 'green', 'lime', 'olive', 'yellow', 'navy', 'blue', 'teal', 'aqua', 'orange', 'aliceblue', 'antiquewhite', 'aquamarine', 'azure', 'beige', 'bisque', 'blanchedalmond', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow', 'limegreen', 'linen', 'magenta', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'oldlace', 'olivedrab', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'whitesmoke', 'yellowgreen', 'rebeccapurple'], UndefinedType]=Undefined, titleFont: Union[str, dict, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, titleFontSize: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, titleFontStyle: Union[str, dict, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, titleFontWeight: Union[dict, '_Parameter', 'SchemaBase', Literal['normal', 'bold', 'lighter', 'bolder', 100, 200, 300, 400, 500, 600, 700, 800, 900], UndefinedType]=Undefined, titleLimit: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, titleLineHeight: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, titleOrient: Union['SchemaBase', Literal['left', 'right', 'top', 'bottom'], UndefinedType]=Undefined, titlePadding: Union[dict, float, '_Parameter', 'SchemaBase', UndefinedType]=Undefined, **kwds):
        super(HeaderConfig, self).__init__(format=format, formatType=formatType, labelAlign=labelAlign, labelAnchor=labelAnchor, labelAngle=labelAngle, labelBaseline=labelBaseline, labelColor=labelColor, labelExpr=labelExpr, labelFont=labelFont, labelFontSize=labelFontSize, labelFontStyle=labelFontStyle, labelFontWeight=labelFontWeight, labelLimit=labelLimit, labelLineHeight=labelLineHeight, labelOrient=labelOrient, labelPadding=labelPadding, labels=labels, orient=orient, title=title, titleAlign=titleAlign, titleAnchor=titleAnchor, titleAngle=titleAngle, titleBaseline=titleBaseline, titleColor=titleColor, titleFont=titleFont, titleFontSize=titleFontSize, titleFontStyle=titleFontStyle, titleFontWeight=titleFontWeight, titleLimit=titleLimit, titleLineHeight=titleLineHeight, titleOrient=titleOrient, titlePadding=titlePadding, **kwds)