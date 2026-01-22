from distutils.core import Command
from distutils.errors import DistutilsSetupError
def _check_rst_data(self, data):
    """Returns warnings when the provided data doesn't compile."""
    source_path = self.distribution.script_name or 'setup.py'
    parser = Parser()
    settings = frontend.OptionParser(components=(Parser,)).get_default_values()
    settings.tab_width = 4
    settings.pep_references = None
    settings.rfc_references = None
    reporter = SilentReporter(source_path, settings.report_level, settings.halt_level, stream=settings.warning_stream, debug=settings.debug, encoding=settings.error_encoding, error_handler=settings.error_encoding_error_handler)
    document = nodes.document(settings, reporter, source=source_path)
    document.note_source(source_path, -1)
    try:
        parser.parse(data, document)
    except AttributeError as e:
        reporter.messages.append((-1, 'Could not finish the parsing: %s.' % e, '', {}))
    return reporter.messages