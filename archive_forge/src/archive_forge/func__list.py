import base64
import json
import random
from tensorboard import errors
from tensorboard.compat.proto import summary_pb2
from tensorboard.data import provider
from tensorboard.util import tb_logging
from tensorboard.util import tensor_util
def _list(self, construct_time_series, index):
    """Helper to list scalar or tensor time series.

        Args:
          construct_time_series: `ScalarTimeSeries` or `TensorTimeSeries`.
          index: The result of `self._index(...)`.

        Returns:
          A list of objects of type given by `construct_time_series`,
          suitable to be returned from `list_scalars` or `list_tensors`.
        """
    result = {}
    for run, tag_to_metadata in index.items():
        result_for_run = {}
        result[run] = result_for_run
        for tag, summary_metadata in tag_to_metadata.items():
            max_step = None
            max_wall_time = None
            for event in self._multiplexer.Tensors(run, tag):
                if max_step is None or max_step < event.step:
                    max_step = event.step
                if max_wall_time is None or max_wall_time < event.wall_time:
                    max_wall_time = event.wall_time
            summary_metadata = self._multiplexer.SummaryMetadata(run, tag)
            result_for_run[tag] = construct_time_series(max_step=max_step, max_wall_time=max_wall_time, plugin_content=summary_metadata.plugin_data.content, description=summary_metadata.summary_description, display_name=summary_metadata.display_name)
    return result