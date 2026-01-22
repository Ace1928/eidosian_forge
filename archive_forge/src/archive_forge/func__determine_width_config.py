import pandas as pd
from mplfinance._arg_validators import _process_kwargs, _validate_vkwargs_dict
def _determine_width_config(xdates, config):
    """
    Given x-axis xdates, and `mpf.plot()` kwargs config,
    determine the widths and linewidths for candles,
    volume bars, ohlc bars, etc.
    """
    datalen = len(xdates)
    avg_dist_between_points = (xdates[-1] - xdates[0]) / float(datalen)
    tweak = 1.06 if datalen > 100 else 1.03
    adjust = tweak * avg_dist_between_points if config['show_nontrading'] else 1.0
    width_config = {}
    if config['width_adjuster_version'] == 'v0':
        width_config['volume_width'] = 0.5 * avg_dist_between_points
        width_config['volume_linewidth'] = None
        width_config['ohlc_ticksize'] = avg_dist_between_points / 2.5
        width_config['ohlc_linewidth'] = None
        width_config['candle_width'] = avg_dist_between_points / 2.0
        width_config['candle_linewidth'] = None
        width_config['line_width'] = None
    else:
        width_config['volume_width'] = _dfinterpolate(_widths, datalen, 'vw') * adjust
        width_config['volume_linewidth'] = _dfinterpolate(_widths, datalen, 'vlw')
        width_config['ohlc_ticksize'] = _dfinterpolate(_widths, datalen, 'ow') * adjust
        width_config['ohlc_linewidth'] = _dfinterpolate(_widths, datalen, 'olw')
        width_config['candle_width'] = _dfinterpolate(_widths, datalen, 'cw') * adjust
        width_config['candle_linewidth'] = _dfinterpolate(_widths, datalen, 'clw')
        width_config['line_width'] = _dfinterpolate(_widths, datalen, 'lw')
    if 'scale_width_adjustment' in config['style']:
        scale = _process_kwargs(config['style']['scale_width_adjustment'], _valid_scale_width_kwargs())
        _scale_width_config(scale, width_config)
    if config['scale_width_adjustment'] is not None:
        scale = _process_kwargs(config['scale_width_adjustment'], _valid_scale_width_kwargs())
        _scale_width_config(scale, width_config)
    if config['update_width_config'] is not None:
        update = _process_kwargs(config['update_width_config'], _valid_update_width_kwargs())
        uplist = [(k, v) for k, v in update.items() if v is not None]
        width_config.update(uplist)
    return width_config