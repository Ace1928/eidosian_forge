from toolz import curried
from ..utils.core import sanitize_dataframe
from ..utils.data import (
from ..utils.data import DataTransformerRegistry as _DataTransformerRegistry
from ..utils.data import DataType, ToValuesReturnType
from ..utils.plugin_registry import PluginEnabler
class DataTransformerRegistry(_DataTransformerRegistry):

    def disable_max_rows(self) -> PluginEnabler:
        """Disable the MaxRowsError."""
        options = self.options
        if self.active in ('default', 'vegafusion'):
            options = options.copy()
            options['max_rows'] = None
        return self.enable(**options)