import json
import decimal
import datetime
import warnings
from pathlib import Path
from plotly.io._utils import validate_coerce_fig_to_dict, validate_coerce_output_type
from _plotly_utils.optional_imports import get_module
from _plotly_utils.basevalidators import ImageUriValidator
from_json_plotly requires a string or bytes argument but received value of type {typ}
def to_json_plotly(plotly_object, pretty=False, engine=None):
    """
    Convert a plotly/Dash object to a JSON string representation

    Parameters
    ----------
    plotly_object:
        A plotly/Dash object represented as a dict, graph_object, or Dash component

    pretty: bool (default False)
        True if JSON representation should be pretty-printed, False if
        representation should be as compact as possible.

    engine: str (default None)
        The JSON encoding engine to use. One of:
          - "json" for an engine based on the built-in Python json module
          - "orjson" for a faster engine that requires the orjson package
          - "auto" for the "orjson" engine if available, otherwise "json"
        If not specified, the default engine is set to the current value of
        plotly.io.json.config.default_engine.

    Returns
    -------
    str
        Representation of input object as a JSON string

    See Also
    --------
    to_json : Convert a plotly Figure to JSON with validation
    """
    orjson = get_module('orjson', should_load=True)
    if engine is None:
        engine = config.default_engine
    if engine == 'auto':
        if orjson is not None:
            engine = 'orjson'
        else:
            engine = 'json'
    elif engine not in ['orjson', 'json']:
        raise ValueError('Invalid json engine: %s' % engine)
    modules = {'sage_all': get_module('sage.all', should_load=False), 'np': get_module('numpy', should_load=False), 'pd': get_module('pandas', should_load=False), 'image': get_module('PIL.Image', should_load=False)}
    if engine == 'json':
        opts = {}
        if pretty:
            opts['indent'] = 2
        else:
            opts['separators'] = (',', ':')
        from _plotly_utils.utils import PlotlyJSONEncoder
        return _safe(json.dumps(plotly_object, cls=PlotlyJSONEncoder, **opts), _swap_json)
    elif engine == 'orjson':
        JsonConfig.validate_orjson()
        opts = orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY
        if pretty:
            opts |= orjson.OPT_INDENT_2
        try:
            plotly_object = plotly_object.to_plotly_json()
        except AttributeError:
            pass
        try:
            return _safe(orjson.dumps(plotly_object, option=opts).decode('utf8'), _swap_orjson)
        except TypeError:
            pass
        cleaned = clean_to_json_compatible(plotly_object, numpy_allowed=True, datetime_allowed=True, modules=modules)
        return _safe(orjson.dumps(cleaned, option=opts).decode('utf8'), _swap_orjson)