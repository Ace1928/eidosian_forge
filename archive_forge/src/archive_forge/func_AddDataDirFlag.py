from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def AddDataDirFlag(parser, emulator_name):
    parser.add_argument('--data-dir', required=False, help="The directory to be used to store/retrieve data/config for an emulator run. The default value is `<USER_CONFIG_DIR>/emulators/{}`. The value of USER_CONFIG_DIR can be found by running:\n\n  $ gcloud info --format='get(config.paths.global_config_dir)'".format(emulator_name))