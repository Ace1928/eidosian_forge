from __future__ import annotations
import logging  # isort:skip
from os.path import basename, join
from docutils import nodes
from docutils.parsers.rst.directives import unchanged
from sphinx.directives.code import CodeBlock, container_wrapper, dedent_lines
from sphinx.errors import SphinxError
from sphinx.locale import __
from sphinx.util import logging, parselinenos
from sphinx.util.nodes import set_source_info
from . import PARALLEL_SAFE
from .templates import (
from .util import get_sphinx_resources
class BokehJSContent(CodeBlock):
    has_content = True
    optional_arguments = 1
    required_arguments = 0
    option_spec = CodeBlock.option_spec
    option_spec.update(title=unchanged)
    option_spec.update(js_file=unchanged)
    option_spec.update(include_html=unchanged)
    option_spec.update(disable_codepen=unchanged)

    def get_codeblock_node(self, code, language):
        """this is copied from sphinx.directives.code.CodeBlock.run

        it has been changed to accept code and language as an arguments instead
        of reading from self

        """
        document = self.state.document
        location = self.state_machine.get_source_and_line(self.lineno)
        linespec = self.options.get('emphasize-lines')
        if linespec:
            try:
                nlines = len(code.split('\n'))
                hl_lines = parselinenos(linespec, nlines)
                if any((i >= nlines for i in hl_lines)):
                    emph_lines = self.options['emphasize-lines']
                    log.warning(__(f'line number spec is out of range(1-{nlines}): {emph_lines!r}'), location=location)
                hl_lines = [x + 1 for x in hl_lines if x < nlines]
            except ValueError as err:
                return [document.reporter.warning(str(err), line=self.lineno)]
        else:
            hl_lines = None
        if 'dedent' in self.options:
            location = self.state_machine.get_source_and_line(self.lineno)
            lines = code.split('\n')
            lines = dedent_lines(lines, self.options['dedent'], location=location)
            code = '\n'.join(lines)
        literal = nodes.literal_block(code, code)
        literal['language'] = language
        literal['linenos'] = 'linenos' in self.options or 'lineno-start' in self.options
        literal['classes'] += self.options.get('class', [])
        extra_args = literal['highlight_args'] = {}
        if hl_lines is not None:
            extra_args['hl_lines'] = hl_lines
        if 'lineno-start' in self.options:
            extra_args['linenostart'] = self.options['lineno-start']
        set_source_info(self, literal)
        caption = self.options.get('caption')
        if caption:
            try:
                literal = container_wrapper(self, literal, caption)
            except ValueError as exc:
                return [document.reporter.warning(str(exc), line=self.lineno)]
        self.add_name(literal)
        return [literal]

    def get_js_source(self):
        js_file = self.options.get('js_file', False)
        if js_file and self.content:
            raise SphinxError("bokehjs-content:: directive can't have both js_file and content")
        if js_file:
            log.debug(f'[bokehjs-content] handling external example in {self.env.docname!r}: {js_file}')
            path = js_file
            if not js_file.startswith('/'):
                path = join(self.env.app.srcdir, path)
            js_source = open(path).read()
        else:
            log.debug(f'[bokehjs-content] handling inline example in {self.env.docname!r}')
            js_source = '\n'.join(self.content)
        return js_source

    def get_code_language(self):
        """
        This is largely copied from bokeh.sphinxext.bokeh_plot.run
        """
        js_source = self.get_js_source()
        if self.options.get('include_html', False):
            resources = get_sphinx_resources(include_bokehjs_api=True)
            html_source = BJS_HTML.render(css_files=resources.css_files, js_files=resources.js_files, hashes=resources.hashes, bjs_script=js_source)
            return [html_source, 'html']
        else:
            return [js_source, 'javascript']

    def run(self):
        rst_source = self.state_machine.node.document['source']
        rst_filename = basename(rst_source)
        serial_no = self.env.new_serialno('ccb')
        target_id = f'{rst_filename}.ccb-{serial_no}'
        target_id = target_id.replace('.', '-')
        target_node = nodes.target('', '', ids=[target_id])
        node = bokehjs_content()
        node['target_id'] = target_id
        node['title'] = self.options.get('title', 'bokehjs example')
        node['include_bjs_header'] = False
        node['disable_codepen'] = self.options.get('disable_codepen', False)
        node['js_source'] = self.get_js_source()
        source_doc = self.state_machine.node.document
        if not hasattr(source_doc, 'bjs_seen'):
            source_doc.bjs_seen = True
            node['include_bjs_header'] = True
        code_content, language = self.get_code_language()
        cb = self.get_codeblock_node(code_content, language)
        node.setup_child(cb[0])
        node.children.append(cb[0])
        return [target_node, node]