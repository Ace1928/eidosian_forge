from bokeh.core.properties import (
from bokeh.events import ModelEvent
from bokeh.models import LayoutDOM
from ..config import config
from ..io.resources import bundled_files
from ..util import classproperty
class EChartsEvent(ModelEvent):
    event_name = 'echarts_event'

    def __init__(self, model, type=None, data=None, query=None):
        self.type = type
        self.data = data
        self.query = query
        super().__init__(model=model)