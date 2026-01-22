import mimetypes
from typing import Optional
import traitlets
from traitlets.config import Config
from nbconvert.filters.markdown_mistune import IPythonRenderer, MarkdownWithMath
from nbconvert.exporters.html import HTMLExporter
from nbconvert.exporters.templateexporter import TemplateExporter
from nbconvert.filters.highlight import Highlight2HTML
from .static_file_handler import TemplateStaticFileHandler
from .utils import create_include_assets_functions
class VoilaExporter(HTMLExporter):
    """Custom HTMLExporter that inlines the images using VoilaMarkdownRenderer"""
    base_url = traitlets.Unicode(help='Base url for resources').tag(config=True)
    markdown_renderer_class = traitlets.Type(VoilaMarkdownRenderer).tag(config=True)
    contents_manager = traitlets.Any()
    show_margins = traitlets.Bool(True, help='show left and right margins for the "lab" template, this gives a "classic" template look').tag(config=True)

    @traitlets.validate('template_name')
    def _validate_template_name(self, template_name):
        if template_name.value == 'classic':
            self.log.warning('"classic" template support will be removed in Voila 1.0.0, please use the "lab" template instead with the "--show-margins" option for a similar look')
        return template_name.value

    @pass_context
    def markdown2html(self, context, source):
        cell = context['cell']
        attachments = cell.get('attachments', {})
        cls = self.markdown_renderer_class
        renderer = cls(escape=False, attachments=attachments, contents_manager=self.contents_manager, anchor_link_text=self.anchor_link_text)
        return MarkdownWithMath(renderer=renderer).render(source)

    @property
    def default_config(self):
        c = Config({'VoilaExporter': {'markdown_renderer_class': 'voila.exporter.VoilaMarkdownRenderer'}})
        c.merge(super(VoilaExporter, self).default_config)
        return c

    @traitlets.default('template_file')
    def default_template_file(self):
        return 'index.html.j2'

    async def generate_from_notebook_node(self, nb, resources=None, extra_context={}, **kw):
        langinfo = nb.metadata.get('language_info', {})
        lexer = langinfo.get('pygments_lexer', langinfo.get('name', None))
        highlight_code = self.filters.get('highlight_code', Highlight2HTML(pygments_lexer=lexer, parent=self))
        self.register_filter('highlight_code', highlight_code)
        nb_copy, resources = super(TemplateExporter, self).from_notebook_node(nb, resources, **kw)
        resources.setdefault('raw_mimetypes', self.raw_mimetypes)
        resources['global_content_filter'] = {'include_code': not self.exclude_code_cell, 'include_markdown': not self.exclude_markdown, 'include_raw': not self.exclude_raw, 'include_unknown': not self.exclude_unknown, 'include_input': not self.exclude_input, 'include_output': not self.exclude_output, 'include_input_prompt': not self.exclude_input_prompt, 'include_output_prompt': not self.exclude_output_prompt, 'no_prompt': self.exclude_input_prompt and self.exclude_output_prompt}
        async for output in self.template.generate_async(nb=nb_copy, resources=resources, **extra_context, static_url=self.static_url):
            yield (output, resources)

    @property
    def environment(self):
        self.enable_async = True
        env = super().environment
        if 'jinja2.ext.do' not in env.extensions:
            env.add_extension('jinja2.ext.do')
        return env

    def get_template_paths(self):
        return self.template_path

    def static_url(self, path):
        """Mimics tornado.web.RequestHandler.static_url"""
        settings = {'static_url_prefix': f'{self.base_url}voila/templates/', 'static_path': None}
        return TemplateStaticFileHandler.make_static_url(settings, f'{self.template_name}/static/{path}')

    def _init_resources(self, resources):
        resources = super(VoilaExporter, self)._init_resources(resources)
        include_assets_functions = create_include_assets_functions(self.template_name, self.base_url)
        resources.update(include_assets_functions)
        resources['show_margins'] = self.show_margins
        return resources