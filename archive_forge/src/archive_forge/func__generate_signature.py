from collections import defaultdict
from ..core import Store
@classmethod
def _generate_signature(cls):
    from inspect import Parameter, Signature
    keywords = ['backend', 'fig', 'holomap', 'widgets', 'fps', 'max_frames', 'size', 'dpi', 'filename', 'info', 'css', 'widget_location']
    return Signature([Parameter(kw, Parameter.KEYWORD_ONLY) for kw in keywords])