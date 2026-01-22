from collections import defaultdict
from ..core import Store
class OutputSettings(KeywordSettings):
    """
    Class for controlling display and output settings.
    """
    allowed = {'backend': list_backends(), 'center': [True, False], 'fig': list_formats('fig'), 'holomap': list_formats('holomap'), 'widgets': ['embed', 'live'], 'fps': (0, float('inf')), 'max_frames': (0, float('inf')), 'max_branches': {None}, 'size': (0, float('inf')), 'dpi': (1, float('inf')), 'filename': {None}, 'info': [True, False], 'widget_location': ['left', 'bottom', 'right', 'top', 'top_left', 'top_right', 'bottom_left', 'bottom_right', 'left_top', 'left_bottom', 'right_top', 'right_bottom'], 'css': {k: str for k in ['width', 'height', 'padding', 'margin', 'max-width', 'min-width', 'max-height', 'min-height', 'outline', 'float']}}
    defaults = dict([('backend', None), ('center', True), ('fig', None), ('holomap', None), ('widgets', None), ('fps', None), ('max_frames', 500), ('size', None), ('dpi', None), ('filename', None), ('info', False), ('widget_location', None), ('css', None)])
    remembered = ['max_frames', 'info', 'filename']
    render_params = ['fig', 'holomap', 'size', 'fps', 'dpi', 'css', 'widget_mode', 'mode', 'widget_location', 'center']
    options = {}
    _backend_options = defaultdict(dict)
    _disable_info_output = False
    last_backend = None
    backend_list = []

    def missing_dependency_exception(value, keyword, allowed):
        raise Exception(f'Format {value!r} does not appear to be supported.')

    def missing_backend_exception(value, keyword, allowed):
        if value in OutputSettings.backend_list:
            raise ValueError(f'Backend {value!r} not available. Has it been loaded with the notebook_extension?')
        else:
            raise ValueError(f'Backend {value!r} does not exist')
    custom_exceptions = {'holomap': missing_dependency_exception, 'backend': missing_backend_exception}
    nbagg_counter = 0

    @classmethod
    def _generate_docstring(cls, signature=False):
        intro = ['Helper used to set HoloViews display options.', 'Arguments are supplied as a series of keywords in any order:', '']
        backend = 'backend      : The backend used by HoloViews'
        fig = 'fig          : The static figure format'
        holomap = 'holomap      : The display type for holomaps'
        widgets = 'widgets      : The widget mode for widgets'
        fps = 'fps          : The frames per second used for animations'
        max_frames = 'max_frames   : The max number of frames rendered (default %r)' % cls.defaults['max_frames']
        size = 'size         : The percentage size of displayed output'
        dpi = 'dpi          : The rendered dpi of the figure'
        filename = 'filename    : The filename of the saved output, if any (default %r)' % cls.defaults['filename']
        info = 'info    : The information to page about the displayed objects (default %r)' % cls.defaults['info']
        css = 'css     : Optional css style attributes to apply to the figure image tag'
        widget_location = 'widget_location : The position of the widgets relative to the plot'
        descriptions = [backend, fig, holomap, widgets, fps, max_frames, size, dpi, filename, info, css, widget_location]
        keywords = ['backend', 'fig', 'holomap', 'widgets', 'fps', 'max_frames', 'size', 'dpi', 'filename', 'info', 'css', 'widget_location']
        if signature:
            doc_signature = '\noutput(%s)\n' % ', '.join(('%s=None' % kw for kw in keywords))
            return '\n'.join([doc_signature] + intro + descriptions)
        else:
            return '\n'.join(intro + descriptions)

    @classmethod
    def _generate_signature(cls):
        from inspect import Parameter, Signature
        keywords = ['backend', 'fig', 'holomap', 'widgets', 'fps', 'max_frames', 'size', 'dpi', 'filename', 'info', 'css', 'widget_location']
        return Signature([Parameter(kw, Parameter.KEYWORD_ONLY) for kw in keywords])

    @classmethod
    def _validate(cls, options, items, warnfn):
        """Validation of edge cases and incompatible options"""
        if 'html' in Store.display_formats:
            pass
        elif 'fig' in items and items['fig'] not in Store.display_formats:
            msg = f'Requesting output figure format {items['fig']!r} ' + f'not in display formats {Store.display_formats!r}'
            if warnfn is None:
                print(f'Warning: {msg}')
            else:
                warnfn(msg)
        backend = Store.current_backend
        return Store.renderers[backend].validate(options)

    @classmethod
    def output(cls, line=None, cell=None, cell_runner=None, help_prompt=None, warnfn=None, **kwargs):
        if line and kwargs:
            raise ValueError('Please either specify a string to parse or keyword arguments')
        elif not Store.renderers:
            raise ValueError('No plotting extension is currently loaded. Ensure you load an plotting extension with hv.extension or import it explicitly from holoviews.plotting before using hv.output.')
        prev_backend = Store.current_backend
        if prev_backend in Store.renderers:
            prev_renderer = Store.renderers[prev_backend]
            prev_backend_spec = prev_backend + ':' + prev_renderer.mode
            prev_params = {k: v for k, v in prev_renderer.param.values().items() if k in cls.render_params}
        else:
            prev_renderer = None
            prev_backend_spec = prev_backend + ':default'
            prev_params = {}
        backend = prev_backend
        prev_restore = dict(OutputSettings.options)
        try:
            if line is not None:
                line = line.split('#')[0].strip()
                kwargs = cls.extract_keywords(line, {})
            options = cls.get_options(kwargs, {}, warnfn)
            if 'backend' in options:
                backend_spec = options['backend']
                if ':' not in backend_spec:
                    backend_spec += ':default'
            else:
                backend_spec = prev_backend_spec
            backend = backend_spec.split(':')[0]
            renderer = Store.renderers[backend]
            render_params = {k: v for k, v in renderer.param.values().items() if k in cls.render_params}
            OutputSettings.options = options
            cls._set_render_options(options, backend_spec)
        except Exception as e:
            OutputSettings.options = prev_restore
            cls.set_backend(prev_backend)
            if backend not in Store.renderers:
                raise ValueError(f'The selected plotting extension {backend!r} has not been loaded, ensure you load it with hv.extension({backend!r}) before using hv.output.') from e
            print(f'Error: {e}')
            if help_prompt:
                print(help_prompt)
            return
        if cell is not None:
            if cell_runner:
                cell_runner(cell, renderer)
            OutputSettings.options = prev_restore
            cls._set_render_options(render_params, backend_spec)
            if backend_spec.split(':')[0] != prev_backend:
                cls.set_backend(prev_backend)
                cls._set_render_options(prev_params, prev_backend_spec)

    @classmethod
    def update_options(cls, options, items):
        """
        Switch default options and backend if new backend is supplied in
        items.
        """
        backend_spec = items.get('backend', Store.current_backend)
        split = backend_spec.split(':')
        backend, mode = split if len(split) == 2 else (split[0], 'default')
        if ':' not in backend_spec:
            backend_spec += ':default'
        if 'max_branches' in items:
            print('Warning: The max_branches option is now deprecated. Ignoring.')
            del items['max_branches']
        prev_backend = Store.current_backend
        renderer = Store.renderers[prev_backend]
        prev_backend_spec = prev_backend + ':' + renderer.mode
        for p in ['fig', 'holomap']:
            cls.allowed[p] = list_formats(p, backend_spec)
        if backend not in Store.renderers:
            options['backend'] = backend_spec
            return options
        backend_options = dict(cls._backend_options[backend_spec])
        cls._backend_options[prev_backend_spec] = {k: v for k, v in cls.options.items() if k in cls.remembered}
        for opt in cls.remembered:
            if opt not in backend_options:
                backend_options[opt] = cls.defaults[opt]
        for p in ['fig', 'holomap']:
            if backend_options.get(p) not in cls.allowed[p]:
                backend_options[p] = cls.allowed[p][0]
        backend_options['backend'] = backend_spec
        backend_options['mode'] = mode
        return backend_options

    @classmethod
    def initialize(cls, backend_list):
        cls.backend_list = backend_list
        backend = Store.current_backend
        if backend in Store.renderers:
            cls.options = dict({k: cls.defaults[k] for k in cls.remembered})
            cls.set_backend(backend)
        else:
            cls.options['backend'] = None
            cls.set_backend(None)

    @classmethod
    def set_backend(cls, backend):
        cls.last_backend = Store.current_backend
        Store.set_current_backend(backend)

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