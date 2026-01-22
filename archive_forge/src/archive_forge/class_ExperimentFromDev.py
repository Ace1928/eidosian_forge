import sys
import time
import grpc
from tensorboard.data.experimental import base_experiment
from tensorboard.data.experimental import utils as experimental_utils
from tensorboard.uploader import auth
from tensorboard.uploader import util
from tensorboard.uploader import server_info as server_info_lib
from tensorboard.uploader.proto import export_service_pb2
from tensorboard.uploader.proto import export_service_pb2_grpc
from tensorboard.uploader.proto import server_info_pb2
from tensorboard.util import grpc_util
class ExperimentFromDev(base_experiment.BaseExperiment):
    """Implementation of BaseExperiment, specialized for tensorboard.dev."""

    def __init__(self, experiment_id, api_endpoint=None):
        """Constructor of ExperimentFromDev.

        Args:
          experiment_id: String ID of the experiment on tensorboard.dev (e.g.,
            "AdYd1TgeTlaLWXx6I8JUbA").
          api_endpoint: Optional override value for API endpoint. Used for
            development only.
        """
        super().__init__()
        self._experiment_id = experiment_id
        self._api_client = get_api_client(api_endpoint=api_endpoint)

    def get_scalars(self, runs_filter=None, tags_filter=None, pivot=False, include_wall_time=False):
        pandas = import_pandas()
        if runs_filter is not None:
            raise NotImplementedError('runs_filter support for get_scalars() is not implemented yet.')
        if tags_filter is not None:
            raise NotImplementedError('tags_filter support for get_scalars() is not implemented yet.')
        request = export_service_pb2.StreamExperimentDataRequest()
        request.experiment_id = self._experiment_id
        read_time = time.time()
        util.set_timestamp(request.read_timestamp, read_time)
        stream = self._api_client.StreamExperimentData(request, metadata=grpc_util.version_metadata())
        runs = []
        tags = []
        steps = []
        wall_times = []
        values = []
        for response in stream:
            num_values = len(response.points.values)
            runs.extend([response.run_name] * num_values)
            tags.extend([response.tag_name] * num_values)
            steps.extend(list(response.points.steps))
            wall_times.extend([t.ToNanoseconds() / 1000000000.0 for t in response.points.wall_times])
            values.extend(list(response.points.values))
        data = {'run': runs, 'tag': tags, 'step': steps, 'value': values}
        if include_wall_time:
            data['wall_time'] = wall_times
        dataframe = pandas.DataFrame(data)
        if pivot:
            dataframe = experimental_utils.pivot_dataframe(dataframe)
        return dataframe