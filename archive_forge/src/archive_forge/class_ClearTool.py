from bokeh.core.properties import Instance, List, Dict, String, Any
from bokeh.models import Tool, ColumnDataSource, PolyEditTool, PolyDrawTool
class ClearTool(Tool):
    """
    Clears the data on the supplied ColumnDataSources.
    """
    sources = List(Instance(ColumnDataSource))