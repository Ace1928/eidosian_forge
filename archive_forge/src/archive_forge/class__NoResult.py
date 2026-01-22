from testtools.matchers import Mismatch
from ._deferred import failure_content, on_deferred_result
class _NoResult:
    """Matches a Deferred that has not yet fired."""

    @staticmethod
    def _got_result(deferred, result):
        return Mismatch('No result expected on %r, found %r instead' % (deferred, result))

    def match(self, deferred):
        """Match ``deferred`` if it hasn't fired."""
        return on_deferred_result(deferred, on_success=self._got_result, on_failure=self._got_result, on_no_result=lambda _: None)