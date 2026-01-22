from .. import urlutils
from ..transport import FileExists
from . import decorator
class BrokenRenameTransportDecorator(decorator.TransportDecorator):
    """A transport that fails to detect clashing renames"""

    @classmethod
    def _get_url_prefix(self):
        """FakeNFS transports are identified by 'brokenrename+'"""
        return 'brokenrename+'

    def rename(self, rel_from, rel_to):
        """See Transport.rename().
        """
        try:
            if self._decorated.has(rel_to):
                rel_to = urlutils.join(rel_to, urlutils.basename(rel_from))
            self._decorated.rename(rel_from, rel_to)
        except (errors.DirectoryNotEmpty, FileExists) as e:
            return