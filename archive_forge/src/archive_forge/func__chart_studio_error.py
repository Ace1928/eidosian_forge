import warnings
import functools
def _chart_studio_error(submodule):
    raise ImportError('\nThe plotly.{submodule} module is deprecated,\nplease install the chart-studio package and use the\nchart_studio.{submodule} module instead. \n'.format(submodule=submodule))