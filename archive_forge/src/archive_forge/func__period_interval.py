import re
from urllib import parse
from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
def _period_interval(self):
    period = self.properties[self.PERIOD]
    if period is None:
        period = self._default_period_interval
    return period