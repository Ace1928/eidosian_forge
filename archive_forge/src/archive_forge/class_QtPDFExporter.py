from traitlets import Bool
from .qt_exporter import QtExporter
class QtPDFExporter(QtExporter):
    """Writer designed to write to PDF files.

    This inherits from :class:`HTMLExporter`. It creates the HTML using the
    template machinery, and then uses pyqtwebengine to create a pdf.
    """
    export_from_notebook = 'PDF via HTML'
    format = 'pdf'
    paginate = Bool(True, help='\n        Split generated notebook into multiple pages.\n\n        If False, a PDF with one long page will be generated.\n\n        Set to True to match behavior of LaTeX based PDF generator\n        ').tag(config=True)