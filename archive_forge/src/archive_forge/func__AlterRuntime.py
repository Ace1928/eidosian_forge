from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
import shutil
import tempfile
from gae_ext_runtime import ext_runtime
from googlecloudsdk.api_lib.app import yaml_parsing
from googlecloudsdk.api_lib.app.runtimes import fingerprinter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.app import deployables
from googlecloudsdk.command_lib.app import exceptions
from googlecloudsdk.command_lib.app import output_helpers
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
from googlecloudsdk.third_party.appengine.api import appinfo
from ruamel import yaml
import six
def _AlterRuntime(config_filename, runtime):
    try:
        with tempfile.NamedTemporaryFile(prefix='app.yaml.') as f:
            backup_fname = f.name
        log.status.Print('Copying original config [{0}] to backup location [{1}].'.format(config_filename, backup_fname))
        shutil.copyfile(config_filename, backup_fname)
        with files.FileReader(config_filename) as yaml_file:
            encoding = yaml_file.encoding
            config = yaml.load(yaml_file, yaml.RoundTripLoader)
        config['runtime'] = runtime
        raw_buf = io.BytesIO()
        tmp_yaml_buf = io.TextIOWrapper(raw_buf, encoding)
        yaml.dump(config, tmp_yaml_buf, Dumper=yaml.RoundTripDumper)
        with files.BinaryFileWriter(config_filename) as yaml_file:
            tmp_yaml_buf.seek(0)
            yaml_file.write(raw_buf.getvalue())
    except Exception as e:
        raise fingerprinter.AlterConfigFileError(e)