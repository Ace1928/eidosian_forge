from ...core.options import Store
from ...core.overlay import NdOverlay, Overlay
from ...selection import OverlaySelectionDisplay
def _style_region_element(self, region_element, unselected_color):
    from ..util import linear_gradient
    backend_options = Store.options(backend='plotly')
    el2_name = None
    if isinstance(region_element, NdOverlay):
        el1_name = type(region_element.last).name
    elif isinstance(region_element, Overlay):
        el1_name = type(region_element.get(0)).name
        el2_name = type(region_element.get(1)).name
    else:
        el1_name = type(region_element).name
    style_options = backend_options[el1_name,]['style']
    allowed_keywords = style_options.allowed_keywords
    options = {}
    if el1_name != 'Histogram':
        if unselected_color:
            region_color = linear_gradient(unselected_color, '#000000', 9)[3]
        if 'Span' in el1_name:
            unselected_color = unselected_color or '#e6e9ec'
            region_color = linear_gradient(unselected_color, '#000000', 9)[1]
        if 'line_width' in allowed_keywords:
            options['line_width'] = 1
    else:
        unselected_color = unselected_color or '#e6e9ec'
        region_color = linear_gradient(unselected_color, '#000000', 9)[1]
    if 'color' in allowed_keywords and unselected_color:
        options['color'] = region_color
    elif 'line_color' in allowed_keywords and unselected_color:
        options['line_color'] = region_color
    if 'selectedpoints' in allowed_keywords:
        options['selectedpoints'] = False
    region = region_element.opts(el1_name, clone=True, backend='plotly', **options)
    if el2_name and el2_name == 'Path':
        region = region.opts(el2_name, backend='plotly', line_color='black')
    return region