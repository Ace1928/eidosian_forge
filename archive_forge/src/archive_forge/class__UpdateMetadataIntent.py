import abc
import os
import sys
import textwrap
from absl import logging
import grpc
from tensorboard.compat import tf
from tensorboard.uploader.proto import experiment_pb2
from tensorboard.uploader.proto import export_service_pb2_grpc
from tensorboard.uploader.proto import write_service_pb2_grpc
from tensorboard.uploader import auth
from tensorboard.uploader import dry_run_stubs
from tensorboard.uploader import exporter as exporter_lib
from tensorboard.uploader import flags_parser
from tensorboard.uploader import formatters
from tensorboard.uploader import server_info as server_info_lib
from tensorboard.uploader import uploader as uploader_lib
from tensorboard.uploader.proto import server_info_pb2
from tensorboard import program
from tensorboard.plugins import base_plugin
class _UpdateMetadataIntent(_Intent):
    """The user intends to update the metadata for an experiment."""
    _MESSAGE_TEMPLATE = textwrap.dedent('        This will modify the metadata associated with the experiment on\n        https://tensorboard.dev with the following experiment ID:\n\n        {experiment_id}\n\n        You have chosen to modify an experiment. All experiments uploaded\n        to TensorBoard.dev are publicly visible. Do not upload sensitive\n        data.\n        ')

    def __init__(self, experiment_id, name=None, description=None):
        self.experiment_id = experiment_id
        self.name = name
        self.description = description

    def get_ack_message_body(self):
        return self._MESSAGE_TEMPLATE.format(experiment_id=self.experiment_id)

    def execute(self, server_info, channel):
        api_client = write_service_pb2_grpc.TensorBoardWriterServiceStub(channel)
        experiment_id = self.experiment_id
        _die_if_bad_experiment_name(self.name)
        _die_if_bad_experiment_description(self.description)
        if not experiment_id:
            raise base_plugin.FlagsError('Must specify a non-empty experiment ID to modify.')
        try:
            uploader_lib.update_experiment_metadata(api_client, experiment_id, name=self.name, description=self.description)
        except uploader_lib.ExperimentNotFoundError:
            _die('No such experiment %s. Either it never existed or it has already been deleted.' % experiment_id)
        except uploader_lib.PermissionDeniedError:
            _die('Cannot modify experiment %s because it is owned by a different user.' % experiment_id)
        except uploader_lib.InvalidArgumentError as e:
            _die('Server cannot modify experiment as requested: %s' % e)
        except grpc.RpcError as e:
            _die('Internal error modifying experiment: %s' % e)
        logging.info('Modified experiment %s.', experiment_id)
        if self.name is not None:
            logging.info('Set name to %r', self.name)
        if self.description is not None:
            logging.info('Set description to %r', repr(self.description))