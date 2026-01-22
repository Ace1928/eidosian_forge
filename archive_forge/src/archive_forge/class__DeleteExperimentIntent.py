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
class _DeleteExperimentIntent(_Intent):
    """The user intends to delete an experiment."""
    _MESSAGE_TEMPLATE = textwrap.dedent('        This will delete the {num} experiment(s) on\n        https://tensorboard.dev with the following experiment ID(s):\n\n        {experiment_id_list}\n\n        You have chosen to delete an experiment. All experiments uploaded\n        to TensorBoard.dev are publicly visible. Do not upload sensitive\n        data.\n        ')

    def __init__(self, experiment_id_list):
        self.experiment_id_list = experiment_id_list

    def get_ack_message_body(self):
        return self._MESSAGE_TEMPLATE.format(num=len(self.experiment_id_list), experiment_id_list=self.experiment_id_list)

    def execute(self, server_info, channel):
        api_client = write_service_pb2_grpc.TensorBoardWriterServiceStub(channel)
        if not self.experiment_id_list:
            raise base_plugin.FlagsError('Must specify at least one experiment ID to delete.')
        results = {}
        NO_ACTION = 'NO_ACTION'
        DIE_ACTION = 'DIE_ACTION'
        for experiment_id in set(self.experiment_id_list):
            if not experiment_id:
                results[experiment_id] = ('Skipping empty experiment_id.', NO_ACTION)
                continue
            try:
                uploader_lib.delete_experiment(api_client, experiment_id)
                results[experiment_id] = ('Deleted experiment %s.' % experiment_id, NO_ACTION)
            except uploader_lib.ExperimentNotFoundError:
                results[experiment_id] = ('No such experiment %s. Either it never existed or it has already been deleted.' % experiment_id, DIE_ACTION)
            except uploader_lib.PermissionDeniedError:
                results[experiment_id] = ('Cannot delete experiment %s because it is owned by a different user.' % experiment_id, DIE_ACTION)
            except grpc.RpcError as e:
                results[experiment_id] = ('Internal error deleting experiment %s: %s.' % (experiment_id, e), DIE_ACTION)
        any_die_action = False
        err_msg = ''
        for msg, action in results.values():
            if action == NO_ACTION:
                print(msg)
            if action == DIE_ACTION:
                err_msg += msg + '\n'
                any_die_action = True
        if any_die_action:
            _die(err_msg)