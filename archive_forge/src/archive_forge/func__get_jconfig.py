import os
import warnings
import pkgutil
from plotly.optional_imports import get_module
from plotly import tools
from ._plotlyjs_version import __plotlyjs_version__
def _get_jconfig(config=None):
    configkeys = ('staticPlot', 'plotlyServerURL', 'editable', 'edits', 'autosizable', 'responsive', 'queueLength', 'fillFrame', 'frameMargins', 'scrollZoom', 'doubleClick', 'showTips', 'showAxisDragHandles', 'showAxisRangeEntryBoxes', 'showLink', 'sendData', 'showSendToCloud', 'linkText', 'showSources', 'displayModeBar', 'modeBarButtonsToRemove', 'modeBarButtonsToAdd', 'modeBarButtons', 'toImageButtonOptions', 'displaylogo', 'watermark', 'plotGlPixelRatio', 'setBackground', 'topojsonURL', 'mapboxAccessToken', 'logging', 'globalTransforms', 'locale', 'locales', 'doubleClickDelay')
    if config and isinstance(config, dict):
        bad_config = [k for k in config if k not in configkeys]
        if bad_config:
            warnings.warn('\nUnrecognized config options supplied: {bad_config}'.format(bad_config=bad_config))
        clean_config = config
    else:
        clean_config = {}
    plotly_platform_url = tools.get_config_plotly_server_url()
    if not clean_config.get('plotlyServerURL', None):
        clean_config['plotlyServerURL'] = plotly_platform_url
    if plotly_platform_url != 'https://plot.ly' and clean_config.get('linkText', None) == 'Export to plot.ly':
        link_domain = plotly_platform_url.replace('https://', '').replace('http://', '')
        link_text = clean_config['linkText'].replace('plot.ly', link_domain)
        clean_config['linkText'] = link_text
    return clean_config