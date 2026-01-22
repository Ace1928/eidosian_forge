from cliff import columns
from osc_lib import utils
class DictColumn(columns.FormattableColumn):
    """Format column for dict content"""

    def human_readable(self):
        return utils.format_dict(self._value)

    def machine_readable(self):
        return dict(self._value or {})