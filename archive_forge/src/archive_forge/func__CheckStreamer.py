from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataproc import storage_helpers
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_attr
def _CheckStreamer(self, poll_result):
    """Checks if need to init a new output streamer.

    Checks if need to init a new output streamer.
    Remote may fail; switch to new output uri.
    Invalidate the streamer instance and init a new one if necessary.

    Args:
      poll_result: Poll result returned from Poll.
    """

    def _PrintEqualsLineAccrossScreen():
        attr = console_attr.GetConsoleAttr()
        log.err.Print('=' * attr.GetTermSize()[0])
    uri = self._GetOutputUri(poll_result)
    if not uri:
        return
    if self.saved_stream_uri and self.saved_stream_uri != uri:
        self.driver_log_streamer = None
        self.saved_stream_uri = None
        _PrintEqualsLineAccrossScreen()
        log.warning("Attempt failed. Streaming new attempt's output.")
        _PrintEqualsLineAccrossScreen()
    if not self.driver_log_streamer:
        self.saved_stream_uri = uri
        self.driver_log_streamer = storage_helpers.StorageObjectSeriesStream(uri)