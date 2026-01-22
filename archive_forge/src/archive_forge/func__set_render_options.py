from collections import defaultdict
from ..core import Store
@classmethod
def _set_render_options(cls, options, backend=None):
    """
        Set options on current Renderer.
        """
    if backend:
        backend = backend.split(':')[0]
    else:
        backend = Store.current_backend
    cls.set_backend(backend)
    if 'widgets' in options:
        options['widget_mode'] = options['widgets']
    renderer = Store.renderers[backend]
    render_options = {k: options[k] for k in cls.render_params if k in options}
    renderer.param.update(**render_options)