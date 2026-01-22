import sys
from traitlets import Dict, default
from .base import get_exporter
from .templateexporter import TemplateExporter
class ScriptExporter(TemplateExporter):
    """A script exporter."""
    _exporters = Dict()
    _lang_exporters = Dict()
    export_from_notebook = 'Script'

    @default('template_file')
    def _template_file_default(self):
        return 'script.j2'

    @default('template_name')
    def _template_name_default(self):
        return 'script'

    def _get_language_exporter(self, lang_name):
        """Find an exporter for the language name from notebook metadata.

        Uses the nbconvert.exporters.script group of entry points.
        Returns None if no exporter is found.
        """
        if lang_name not in self._lang_exporters:
            try:
                exporters = entry_points(group='nbconvert.exporters.script')
                exporter = [e for e in exporters if e.name == lang_name][0].load()
            except (KeyError, IndexError):
                self._lang_exporters[lang_name] = None
            else:
                self._lang_exporters[lang_name] = exporter(config=self.config, parent=self)
        return self._lang_exporters[lang_name]

    def from_notebook_node(self, nb, resources=None, **kw):
        """Convert from notebook node."""
        langinfo = nb.metadata.get('language_info', {})
        exporter_name = langinfo.get('nbconvert_exporter')
        if exporter_name and exporter_name != 'script':
            self.log.debug('Loading script exporter: %s', exporter_name)
            if exporter_name not in self._exporters:
                exporter = get_exporter(exporter_name)
                self._exporters[exporter_name] = exporter(config=self.config, parent=self)
            exporter = self._exporters[exporter_name]
            return exporter.from_notebook_node(nb, resources, **kw)
        lang_name = langinfo.get('name')
        if lang_name:
            self.log.debug('Using script exporter for language: %s', lang_name)
            exporter = self._get_language_exporter(lang_name)
            if exporter is not None:
                return exporter.from_notebook_node(nb, resources, **kw)
        self.file_extension = langinfo.get('file_extension', '.txt')
        self.output_mimetype = langinfo.get('mimetype', 'text/plain')
        return super().from_notebook_node(nb, resources, **kw)