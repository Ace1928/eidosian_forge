from pathlib import Path
from .config_args import default_config_file, load_config_from_file
from .config_utils import SubcommandHelpFormatter
def update_config_command(args):
    config_file = update_config(args)
    print(f'Sucessfully updated the configuration file at {config_file}.')