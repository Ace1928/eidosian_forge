import six
from genshi.input import ET, HTML, XML
from genshi.output import DocType
from genshi.template.base import Template
from genshi.template.loader import TemplateLoader
from genshi.template.markup import MarkupTemplate
from genshi.template.text import TextTemplate, NewTextTemplate
class AbstractTemplateEnginePlugin(object):
    """Implementation of the plugin API."""
    template_class = None
    extension = None

    def __init__(self, extra_vars_func=None, options=None):
        self.get_extra_vars = extra_vars_func
        if options is None:
            options = {}
        self.options = options
        self.default_encoding = options.get('genshi.default_encoding', None)
        auto_reload = options.get('genshi.auto_reload', '1')
        if isinstance(auto_reload, six.string_types):
            auto_reload = auto_reload.lower() in ('1', 'on', 'yes', 'true')
        search_path = [p for p in options.get('genshi.search_path', '').split(':') if p]
        self.use_package_naming = not search_path
        try:
            max_cache_size = int(options.get('genshi.max_cache_size', 25))
        except ValueError:
            raise ConfigurationError('Invalid value for max_cache_size: "%s"' % options.get('genshi.max_cache_size'))
        loader_callback = options.get('genshi.loader_callback', None)
        if loader_callback and (not hasattr(loader_callback, '__call__')):
            raise ConfigurationError('loader callback must be a function')
        lookup_errors = options.get('genshi.lookup_errors', 'strict')
        if lookup_errors not in ('lenient', 'strict'):
            raise ConfigurationError('Unknown lookup errors mode "%s"' % lookup_errors)
        try:
            allow_exec = bool(options.get('genshi.allow_exec', True))
        except ValueError:
            raise ConfigurationError('Invalid value for allow_exec "%s"' % options.get('genshi.allow_exec'))
        self.loader = TemplateLoader([p for p in search_path if p], auto_reload=auto_reload, max_cache_size=max_cache_size, default_class=self.template_class, variable_lookup=lookup_errors, allow_exec=allow_exec, callback=loader_callback)

    def load_template(self, templatename, template_string=None):
        """Find a template specified in python 'dot' notation, or load one from
        a string.
        """
        if template_string is not None:
            return self.template_class(template_string)
        if self.use_package_naming:
            divider = templatename.rfind('.')
            if divider >= 0:
                from pkg_resources import resource_filename
                package = templatename[:divider]
                basename = templatename[divider + 1:] + self.extension
                templatename = resource_filename(package, basename)
        return self.loader.load(templatename)

    def _get_render_options(self, format=None, fragment=False):
        if format is None:
            format = self.default_format
        kwargs = {'method': format}
        if self.default_encoding:
            kwargs['encoding'] = self.default_encoding
        return kwargs

    def render(self, info, format=None, fragment=False, template=None):
        """Render the template to a string using the provided info."""
        kwargs = self._get_render_options(format=format, fragment=fragment)
        return self.transform(info, template).render(**kwargs)

    def transform(self, info, template):
        """Render the output to an event stream."""
        if not isinstance(template, Template):
            template = self.load_template(template)
        return template.generate(**info)