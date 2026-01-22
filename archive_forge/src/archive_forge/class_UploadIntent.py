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
class UploadIntent(_Intent):
    """The user intends to upload an experiment from the given logdir."""
    _MESSAGE_TEMPLATE = textwrap.dedent('        This will upload your TensorBoard logs to https://tensorboard.dev/ from\n        the following directory:\n\n        {logdir}\n\n        This TensorBoard will be visible to everyone. Do not upload sensitive\n        data.\n        ')

    def __init__(self, logdir, name=None, description=None, verbosity=None, dry_run=None, one_shot=None, experiment_url_callback=None):
        self.logdir = logdir
        self.name = name
        self.description = description
        self.verbosity = verbosity
        self.dry_run = False if dry_run is None else dry_run
        self.one_shot = False if one_shot is None else one_shot
        self.experiment_url_callback = experiment_url_callback

    def get_ack_message_body(self):
        return self._MESSAGE_TEMPLATE.format(logdir=self.logdir)

    def execute(self, server_info, channel):
        if self.dry_run:
            api_client = dry_run_stubs.DryRunTensorBoardWriterStub()
        else:
            api_client = write_service_pb2_grpc.TensorBoardWriterServiceStub(channel)
        _die_if_bad_experiment_name(self.name)
        _die_if_bad_experiment_description(self.description)
        uploader = uploader_lib.TensorBoardUploader(api_client, self.logdir, allowed_plugins=server_info_lib.allowed_plugins(server_info), upload_limits=server_info_lib.upload_limits(server_info), name=self.name, description=self.description, verbosity=self.verbosity, one_shot=self.one_shot)
        if self.one_shot and (not tf.io.gfile.isdir(self.logdir)):
            print('%s: No such directory.' % self.logdir)
            print('User specified `one_shot` mode with an unavailable logdir. Exiting without creating an experiment.')
            return
        experiment_id = uploader.create_experiment()
        url = server_info_lib.experiment_url(server_info, experiment_id)
        if self.experiment_url_callback is not None:
            self.experiment_url_callback(url)
        if not self.one_shot:
            print("Upload started and will continue reading any new data as it's added to the logdir.\n\nTo stop uploading, press Ctrl-C.")
        if self.dry_run:
            print('\n** This is a dry run. No data will be sent to tensorboard.dev. **\n')
        else:
            print('\nNew experiment created. View your TensorBoard at: %s\n' % url)
        interrupted = False
        try:
            uploader.start_uploading()
        except uploader_lib.ExperimentNotFoundError:
            print('Experiment was deleted; uploading has been cancelled')
            return
        except KeyboardInterrupt:
            interrupted = True
        finally:
            if self.one_shot and (not uploader.has_data()):
                print('TensorBoard was run in `one_shot` mode, but did not find any uploadable data in the specified logdir: %s\nAn empty experiment was created. To delete the empty experiment you can execute the following\n\n    tensorboard dev delete --experiment_id=%s' % (self.logdir, uploader.experiment_id))
            end_message = '\n\n'
            if interrupted:
                end_message += 'Interrupted.'
            else:
                end_message += 'Done.'
            if not self.dry_run and uploader.has_data():
                end_message += ' View your TensorBoard at %s' % url
            sys.stdout.write(end_message + '\n')
            sys.stdout.flush()