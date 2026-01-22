from twisted.application.app import ReactorSelectionMixin
from twisted.python.filepath import FilePath
from twisted.python.usage import Options
from twisted.scripts.trial import _BasicOptions
class WorkerOptions(_BasicOptions, Options, ReactorSelectionMixin):
    """
    Options forwarded to the trial distributed worker.
    """

    def coverdir(self):
        """
        Return a L{FilePath} representing the directory into which coverage
        results should be written.
        """
        return FilePath('coverage')