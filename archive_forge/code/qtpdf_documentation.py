from traitlets import Bool
from .qt_exporter import QtExporter
Writer designed to write to PDF files.

    This inherits from :class:`HTMLExporter`. It creates the HTML using the
    template machinery, and then uses pyqtwebengine to create a pdf.
    