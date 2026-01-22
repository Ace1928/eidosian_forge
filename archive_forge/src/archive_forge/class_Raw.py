import sys
import os.path
import re
import time
from docutils import io, nodes, statemachine, utils
from docutils.utils.error_reporting import SafeString, ErrorString
from docutils.utils.error_reporting import locale_encoding
from docutils.parsers.rst import Directive, convert_directive_function
from docutils.parsers.rst import directives, roles, states
from docutils.parsers.rst.directives.body import CodeBlock, NumberLines
from docutils.parsers.rst.roles import set_classes
from docutils.transforms import misc
class Raw(Directive):
    """
    Pass through content unchanged

    Content is included in output based on type argument

    Content may be included inline (content section of directive) or
    imported from a file or url.
    """
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {'file': directives.path, 'url': directives.uri, 'encoding': directives.encoding}
    has_content = True

    def run(self):
        if not self.state.document.settings.raw_enabled or (not self.state.document.settings.file_insertion_enabled and ('file' in self.options or 'url' in self.options)):
            raise self.warning('"%s" directive disabled.' % self.name)
        attributes = {'format': ' '.join(self.arguments[0].lower().split())}
        encoding = self.options.get('encoding', self.state.document.settings.input_encoding)
        e_handler = self.state.document.settings.input_encoding_error_handler
        if self.content:
            if 'file' in self.options or 'url' in self.options:
                raise self.error('"%s" directive may not both specify an external file and have content.' % self.name)
            text = '\n'.join(self.content)
        elif 'file' in self.options:
            if 'url' in self.options:
                raise self.error('The "file" and "url" options may not be simultaneously specified for the "%s" directive.' % self.name)
            source_dir = os.path.dirname(os.path.abspath(self.state.document.current_source))
            path = os.path.normpath(os.path.join(source_dir, self.options['file']))
            path = utils.relative_path(None, path)
            try:
                raw_file = io.FileInput(source_path=path, encoding=encoding, error_handler=e_handler)
                self.state.document.settings.record_dependencies.add(path)
            except IOError as error:
                raise self.severe('Problems with "%s" directive path:\n%s.' % (self.name, ErrorString(error)))
            try:
                text = raw_file.read()
            except UnicodeError as error:
                raise self.severe('Problem with "%s" directive:\n%s' % (self.name, ErrorString(error)))
            attributes['source'] = path
        elif 'url' in self.options:
            source = self.options['url']
            import urllib.request, urllib.error, urllib.parse
            try:
                raw_text = urllib.request.urlopen(source).read()
            except (urllib.error.URLError, IOError, OSError) as error:
                raise self.severe('Problems with "%s" directive URL "%s":\n%s.' % (self.name, self.options['url'], ErrorString(error)))
            raw_file = io.StringInput(source=raw_text, source_path=source, encoding=encoding, error_handler=e_handler)
            try:
                text = raw_file.read()
            except UnicodeError as error:
                raise self.severe('Problem with "%s" directive:\n%s' % (self.name, ErrorString(error)))
            attributes['source'] = source
        else:
            self.assert_has_content()
        raw_node = nodes.raw('', text, **attributes)
        raw_node.source, raw_node.line = self.state_machine.get_source_and_line(self.lineno)
        return [raw_node]