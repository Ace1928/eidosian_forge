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
class TensorBoardExporter:
    """Exports all of the user's experiment data from TensorBoard.dev.

    Data is exported into a directory, with one file per experiment. Each
    experiment file is a sequence of time series, represented as a stream
    of JSON objects, one per line. Each JSON object includes a run name,
    tag name, `tensorboard.compat.proto.summary_pb2.SummaryMetadata` proto
    (base64-encoded, standard RFC 4648 alphabet), and set of points.
    Points are stored in three equal-length lists of steps, wall times (as
    seconds since epoch), and scalar values, for storage efficiency.

    Such streams of JSON objects may be conveniently processed with tools
    like jq(1).

    For example one line of an experiment file might read (when
    pretty-printed):

        {
          "points": {
            "steps": [0, 5],
            "values": [4.8935227394104, 2.5438034534454346],
            "wall_times": [1563406522.669238, 1563406523.0268838]
          },
          "run": "lr_1E-04,conv=1,fc=2",
          "summary_metadata": "CgkKB3NjYWxhcnMSC3hlbnQveGVudF8x",
          "tag": "xent/xent_1"
        }

    This is a time series with two points, both logged on 2019-07-17, one
    about 0.36 seconds after the other.
    """

    def __init__(self, reader_service_client, output_directory):
        """Constructs a TensorBoardExporter.

        Args:
          reader_service_client: A TensorBoardExporterService stub instance.
          output_directory: Path to a directory into which to write data. The
            directory must not exist, to avoid stomping existing or concurrent
            output. Its ancestors will be created if needed.
        """
        self._api = reader_service_client
        self._outdir = output_directory
        parent_dir = os.path.dirname(self._outdir)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        try:
            os.mkdir(self._outdir)
        except OSError as e:
            if e.errno == errno.EEXIST:
                raise OutputDirectoryExistsError()

    def export(self, read_time=None):
        """Executes the export flow.

        Args:
          read_time: A fixed timestamp from which to export data, as float seconds
            since epoch (like `time.time()`). Optional; defaults to the current
            time.

        Yields:
          After each experiment is successfully downloaded, the ID of that
          experiment, as a string.
        """
        if read_time is None:
            read_time = time.time()
        experiment_metadata_mask = experiment_pb2.ExperimentMask(create_time=True, update_time=True, name=True, description=True)
        experiments = list_experiments(self._api, fieldmask=experiment_metadata_mask, read_time=read_time)
        for experiment in experiments:
            experiment_id = experiment.experiment_id
            experiment_metadata = {'name': experiment.name, 'description': experiment.description, 'create_time': util.format_time_absolute(experiment.create_time), 'update_time': util.format_time_absolute(experiment.update_time)}
            experiment_dir = _experiment_directory(self._outdir, experiment_id)
            os.mkdir(experiment_dir)
            metadata_filepath = os.path.join(experiment_dir, _FILENAME_METADATA)
            with open(metadata_filepath, 'x') as outfile:
                json.dump(experiment_metadata, outfile, sort_keys=True)
                outfile.write('\n')
            try:
                data = self._request_json_data(experiment_id, read_time)
                with contextlib.ExitStack() as stack:
                    file_handles = {filename: stack.enter_context(open(os.path.join(experiment_dir, filename), 'x')) for filename in (_FILENAME_SCALARS, _FILENAME_TENSORS, _FILENAME_BLOB_SEQUENCES)}
                    os.mkdir(os.path.join(experiment_dir, _DIRNAME_TENSORS))
                    os.mkdir(os.path.join(experiment_dir, _DIRNAME_BLOBS))
                    for block, filename in data:
                        outfile = file_handles[filename]
                        json.dump(block, outfile, sort_keys=True)
                        outfile.write('\n')
                        outfile.flush()
                    yield experiment_id
            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.CANCELLED:
                    raise GrpcTimeoutException(experiment_id)
                else:
                    raise

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

    def _process_scalar_points(self, points):
        """Process scalar data points.

        Args:
          points: `export_service_pb2.StreamExperimentDataResponse.ScalarPoints`
            proto.

        Returns:
          A JSON-serializable `dict` for the steps, wall_times and values of the
            scalar data points.
        """
        wall_times = [t.ToNanoseconds() / 1000000000.0 for t in points.wall_times]
        return {'steps': list(points.steps), 'wall_times': wall_times, 'values': list(points.values)}

    def _process_tensor_points(self, points, experiment_id):
        """Process tensor data points.

        Args:
          points: `export_service_pb2.StreamExperimentDataResponse.TensorPoints`
            proto.
          experiment_id: ID of the experiment that the `TensorPoints` is a part
            of.

        Returns:
          A JSON-serializable `dict` for the steps, wall_times and the path to
            the .npz files that contain the saved tensor values.
        """
        wall_times = [t.ToNanoseconds() / 1000000000.0 for t in points.wall_times]
        json_object = {'steps': list(points.steps), 'wall_times': wall_times, 'tensors_file_path': None}
        if not json_object['steps']:
            return json_object
        experiment_dir = _experiment_directory(self._outdir, experiment_id)
        tensors_file_path = self._get_tensor_file_path(experiment_dir, json_object['wall_times'][0])
        ndarrays = [tensor_util.make_ndarray(tensor_proto) for tensor_proto in points.values]
        ndarrays = [self._fix_string_types(x) for x in ndarrays]
        np.savez(os.path.join(experiment_dir, tensors_file_path), *ndarrays)
        json_object['tensors_file_path'] = tensors_file_path
        return json_object

    def _fix_string_types(self, ndarray):
        """Change the dtype of text arrays to String rather than Object.

        np.savez ends up pickling np.object arrays, while it doesn't pickle
        strings.  The downside is that it needs to pad the length of each string
        in the array to the maximal length string.  We only want to do this
        type override in this final step of the export path.

        Args:
          ndarray: a tensor converted to an ndarray

        Returns:
          The original ndarray if not np.object, dtype converted to String
          if np.object.
        """
        if ndarray.dtype != np.object_:
            return ndarray
        else:
            return ndarray.astype('|S')

    def _get_tensor_file_path(self, experiment_dir, wall_time):
        """Get a nonexistent path for a tensor value.

        Args:
          experiment_dir: Experiment directory.
          wall_time: Timestamp of the tensor (seconds since the epoch in double).

        Returns:
          A nonexistent path for the tensor, relative to the experiemnt_dir.
        """
        index = 0
        while True:
            tensor_file_path = os.path.join(_DIRNAME_TENSORS, '%.6f' % wall_time + ('_%d' % index if index else '') + '.npz')
            if not os.path.exists(os.path.join(experiment_dir, tensor_file_path)):
                return tensor_file_path
            index += 1

    def _process_blob_sequence_points(self, blob_sequences, experiment_id):
        """Process blob sequence points.

        As a side effect, also downloads the binary contents of the blobs
        to respective files. The paths to the files relative to the
        experiment directory is encapsulated in the returned JSON object.

        Args:
          blob_sequences:
            `export_service_pb2.StreamDataResponse.BlobSequencePoints` proto.

        Returns:
          A JSON-serializable `dict` for the steps and wall_times, as well as
            the blob_file_paths, which are the relative paths to the downloaded
            blob contents.
        """
        wall_times = [t.ToNanoseconds() / 1000000000.0 for t in blob_sequences.wall_times]
        json_object = {'steps': list(blob_sequences.steps), 'wall_times': wall_times, 'blob_file_paths': []}
        blob_file_paths = json_object['blob_file_paths']
        for blobseq in blob_sequences.values:
            seq_blob_file_paths = []
            for entry in blobseq.entries:
                if entry.blob.state == blob_pb2.BlobState.BLOB_STATE_CURRENT:
                    blob_path = self._download_blob(entry.blob.blob_id, experiment_id)
                    seq_blob_file_paths.append(blob_path)
                else:
                    seq_blob_file_paths.append(None)
            blob_file_paths.append(seq_blob_file_paths)
        return json_object

    def _download_blob(self, blob_id, experiment_id):
        """Download the blob via rpc.

        Args:
          blob_id: Id of the blob.
          experiment_id: Id of the experiment that the blob belongs to.

        Returns:
          If the blob is downloaded successfully:
            The path of the downloaded blob file relative to the experiment
            directory.
          Else:
            `None`.
        """
        experiment_dir = _experiment_directory(self._outdir, experiment_id)
        request = export_service_pb2.StreamBlobDataRequest(blob_id=blob_id)
        blob_abspath = os.path.join(experiment_dir, _DIRNAME_BLOBS, _FILENAME_BLOBS_PREFIX + blob_id + _FILENAME_BLOBS_SUFFIX)
        with open(blob_abspath, 'xb') as f:
            try:
                for response in self._api.StreamBlobData(request, metadata=grpc_util.version_metadata()):
                    f.write(response.data)
            except grpc.RpcError as rpc_error:
                logger.error('Omitting blob (id: %s) due to download failure: %s', blob_id, rpc_error)
                return None
        return os.path.relpath(blob_abspath, experiment_dir)