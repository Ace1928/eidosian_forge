from statistics import mean
import geopandas
from shapely.geometry import LineString
import numpy as np
import pandas as pd
from packaging.version import Version
def _categorical_legend(m, title, categories, colors):
    """
    Add categorical legend to a map

    The implementation is using the code originally written by Michel Metran
    (@michelmetran) and released on GitHub
    (https://github.com/michelmetran/package_folium) under MIT license.

    Copyright (c) 2020 Michel Metran

    Parameters
    ----------
    m : folium.Map
        Existing map instance on which to draw the plot
    title : str
        title of the legend (e.g. column name)
    categories : list-like
        list of categories
    colors : list-like
        list of colors (in the same order as categories)
    """
    head = '\n    {% macro header(this, kwargs) %}\n    <script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"></script>\n    <script>$( function() {\n        $( ".maplegend" ).draggable({\n            start: function (event, ui) {\n                $(this).css({\n                    right: "auto",\n                    top: "auto",\n                    bottom: "auto"\n                });\n            }\n        });\n    });\n    </script>\n    <style type=\'text/css\'>\n      .maplegend {\n        position: absolute;\n        z-index:9999;\n        background-color: rgba(255, 255, 255, .8);\n        border-radius: 5px;\n        box-shadow: 0 0 15px rgba(0,0,0,0.2);\n        padding: 10px;\n        font: 12px/14px Arial, Helvetica, sans-serif;\n        right: 10px;\n        bottom: 20px;\n      }\n      .maplegend .legend-title {\n        text-align: left;\n        margin-bottom: 5px;\n        font-weight: bold;\n        }\n      .maplegend .legend-scale ul {\n        margin: 0;\n        margin-bottom: 0px;\n        padding: 0;\n        float: left;\n        list-style: none;\n        }\n      .maplegend .legend-scale ul li {\n        list-style: none;\n        margin-left: 0;\n        line-height: 16px;\n        margin-bottom: 2px;\n        }\n      .maplegend ul.legend-labels li span {\n        display: block;\n        float: left;\n        height: 14px;\n        width: 14px;\n        margin-right: 5px;\n        margin-left: 0;\n        border: 0px solid #ccc;\n        }\n      .maplegend .legend-source {\n        color: #777;\n        clear: both;\n        }\n      .maplegend a {\n        color: #777;\n        }\n    </style>\n    {% endmacro %}\n    '
    import branca as bc
    macro = bc.element.MacroElement()
    macro._template = bc.element.Template(head)
    m.get_root().add_child(macro)
    body = f"\n    <div id='maplegend {title}' class='maplegend'>\n        <div class='legend-title'>{title}</div>\n        <div class='legend-scale'>\n            <ul class='legend-labels'>"
    for label, color in zip(categories, colors):
        body += f"\n                <li><span style='background:{color}'></span>{label}</li>"
    body += '\n            </ul>\n        </div>\n    </div>\n    '
    body = bc.element.Element(body, 'legend')
    m.get_root().html.add_child(body)