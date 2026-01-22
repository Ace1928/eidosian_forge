import sys
from _pydevd_bundle.pydevd_constants import PANDAS_MAX_ROWS, PANDAS_MAX_COLS, PANDAS_MAX_COLWIDTH
from _pydevd_bundle.pydevd_extension_api import TypeResolveProvider, StrPresentationProvider
from _pydevd_bundle.pydevd_resolver import inspect, MethodWrapperType
from _pydevd_bundle.pydevd_utils import Timer
from .pydevd_helpers import find_mod_attr
from contextlib import contextmanager
class PandasSeriesTypeResolveProvider(object):

    def can_provide(self, type_object, type_name):
        series_class = find_mod_attr('pandas.core.series', 'Series')
        return series_class is not None and issubclass(type_object, series_class)

    def resolve(self, obj, attribute):
        return getattr(obj, attribute)

    def get_dictionary(self, obj):
        replacements = {'T': '<transposed dataframe -- debugger:skipped eval>', '_series': '<dict[index:Series] -- debugger:skipped eval>', 'style': '<pandas.io.formats.style.Styler -- debugger: skipped eval>'}
        return _get_dictionary(obj, replacements)

    def get_str_in_context(self, df, context: str):
        """
        :param context:
            This is the context in which the variable is being requested. Valid values:
                "watch",
                "repl",
                "hover",
                "clipboard"
        """
        if context in ('repl', 'clipboard'):
            return repr(df)
        return self.get_str(df)

    def get_str(self, series):
        with customize_pandas_options():
            return repr(series)