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
class _ExportIntent(_Intent):
    """The user intends to download all their experiment data."""
    _MESSAGE_TEMPLATE = textwrap.dedent('        This will download all your experiment data from https://tensorboard.dev\n        and save it to the following directory:\n\n        {output_dir}\n\n        Downloading your experiment data does not delete it from the\n        service. All experiments uploaded to TensorBoard.dev are publicly\n        visible. Do not upload sensitive data.\n        ')

    def __init__(self, output_dir):
        self.output_dir = output_dir

    def get_ack_message_body(self):
        return self._MESSAGE_TEMPLATE.format(output_dir=self.output_dir)

    def execute(self, server_info, channel):
        api_client = export_service_pb2_grpc.TensorBoardExporterServiceStub(channel)
        outdir = self.output_dir
        try:
            exporter = exporter_lib.TensorBoardExporter(api_client, outdir)
        except exporter_lib.OutputDirectoryExistsError:
            msg = 'Output directory already exists: %r' % outdir
            raise base_plugin.FlagsError(msg)
        num_experiments = 0
        try:
            for experiment_id in exporter.export():
                num_experiments += 1
                print('Downloaded experiment %s' % experiment_id)
        except exporter_lib.GrpcTimeoutException as e:
            print('\nUploader has failed because of a timeout error.  Please reach out via e-mail to tensorboard.dev-support@google.com to get help completing your export of experiment %s.' % e.experiment_id)
        print('Done. Downloaded %d experiments to: %s' % (num_experiments, outdir))