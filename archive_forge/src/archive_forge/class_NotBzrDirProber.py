from .. import controldir, errors, tests, ui
from .scenarios import load_tests_apply_scenarios
class NotBzrDirProber(controldir.Prober):

    def probe_transport(self, transport):
        """Our format is present if the transport ends in '.not/'."""
        if transport.has('.not'):
            return NotBzrDirFormat()

    @classmethod
    def known_formats(cls):
        return [NotBzrDirFormat()]