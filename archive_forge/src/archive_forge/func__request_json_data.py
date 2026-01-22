import base64
import contextlib
import errno
import grpc
import json
import os
import string
import time
import numpy as np
from tensorboard.uploader.proto import blob_pb2
from tensorboard.uploader.proto import experiment_pb2
from tensorboard.uploader.proto import export_service_pb2
from tensorboard.uploader import util
from tensorboard.util import grpc_util
from tensorboard.util import tb_logging
from tensorboard.util import tensor_util
def _request_json_data(self, experiment_id, read_time):
    """Given experiment id, generates JSON data and destination file name.

        The JSON data describes the run, tag, metadata, in addition to
          - Actual data in the case of scalars
          - Pointer to binary files in the case of blob sequences.

        For the case of blob sequences, this method has the side effect of
          downloading the contents of the blobs and writing them to files in
          a subdirectory of the experiment directory.

        Args:
          experiment_id: The id of the experiment to request data for.
          read_time: A fixed timestamp from which to export data, as float
            seconds since epoch (like `time.time()`). Optional; defaults to the
            current time.

        Yields:
          (JSON-serializable data, destination file name) tuples.
        """
    request = export_service_pb2.StreamExperimentDataRequest()
    request.experiment_id = experiment_id
    util.set_timestamp(request.read_timestamp, read_time)
    stream = self._api.StreamExperimentData(request, metadata=grpc_util.version_metadata())
    for response in stream:
        metadata = base64.b64encode(response.tag_metadata.SerializeToString()).decode('ascii')
        json_data = {'run': response.run_name, 'tag': response.tag_name, 'summary_metadata': metadata}
        filename = None
        if response.HasField('points'):
            json_data['points'] = self._process_scalar_points(response.points)
            filename = _FILENAME_SCALARS
        elif response.HasField('tensors'):
            json_data['points'] = self._process_tensor_points(response.tensors, experiment_id)
            filename = _FILENAME_TENSORS
        elif response.HasField('blob_sequences'):
            json_data['points'] = self._process_blob_sequence_points(response.blob_sequences, experiment_id)
            filename = _FILENAME_BLOB_SEQUENCES
        if filename:
            yield (json_data, filename)