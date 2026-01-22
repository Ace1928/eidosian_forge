from bokeh.core.properties import Instance, List, Dict, String, Any
from bokeh.models import Tool, ColumnDataSource, PolyEditTool, PolyDrawTool
class RestoreTool(Tool):
    """
    Restores the data on the supplied ColumnDataSources to a previous
    checkpoint created by the CheckpointTool
    """
    sources = List(Instance(ColumnDataSource))