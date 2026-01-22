import socket
from .base import StatsClientBase, PipelineBase
class StreamPipeline(PipelineBase):

    def _send(self):
        self._client._after('\n'.join(self._stats))
        self._stats.clear()