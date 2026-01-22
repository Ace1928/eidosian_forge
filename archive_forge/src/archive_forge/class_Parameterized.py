import weakref
from functools import partial
import bokeh
import bokeh.core.properties as bp
import param as pm
from bokeh.model import DataModel
from bokeh.models import ColumnDataSource
from ..reactive import Syncable
from .document import unlocked
from .notebook import push
from .state import state
class Parameterized(bokeh.core.property.bases.Property):
    """ Accept a Parameterized object.

    This property only exists to support type validation, e.g. for "accepts"
    clauses. It is not serializable itself, and is not useful to add to
    Bokeh models directly.

    """

    def validate(self, value, detail=True):
        super().validate(value, detail)
        if isinstance(value, pm.Parameterized):
            return
        msg = '' if not detail else f'expected param.Parameterized, got {value!r}'
        raise ValueError(msg)