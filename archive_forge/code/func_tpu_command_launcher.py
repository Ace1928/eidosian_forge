import argparse
import os
import subprocess
from packaging.version import Version, parse
from accelerate.commands.config.config_args import default_config_file, load_config_from_file
def tpu_command_launcher(args):
    defaults = None
    if args.config_file is not None or os.path.isfile(default_config_file):
        defaults = load_config_from_file(args.config_file)
        if not args.command_file and defaults.command_file is not None and (not args.command):
            args.command_file = defaults.command_file
        if not args.command and defaults.commands is not None:
            args.command = defaults.commands
        if not args.tpu_name:
            args.tpu_name = defaults.tpu_name
        if not args.tpu_zone:
            args.tpu_zone = defaults.tpu_zone
    if args.accelerate_version == 'dev':
        args.accelerate_version = 'git+https://github.com/huggingface/accelerate.git'
    elif args.accelerate_version == 'latest':
        args.accelerate_version = 'accelerate -U'
    elif isinstance(parse(args.accelerate_version), Version):
        args.accelerate_version = f'accelerate=={args.accelerate_version}'
    if not args.command_file and (not args.command):
        raise ValueError('You must specify either a command file or a command to run on the pod.')
    if args.command_file:
        with open(args.command_file) as f:
            args.command = [f.read().splitlines()]
    if isinstance(args.command[0], list):
        args.command = [line for cmd in args.command for line in cmd]
    new_cmd = ['cd /usr/share']
    if args.install_accelerate:
        new_cmd += [f'pip install {args.accelerate_version}']
    new_cmd += args.command
    args.command = '; '.join(new_cmd)
    cmd = ['gcloud']
    if args.use_alpha:
        cmd += ['alpha']
    cmd += ['compute', 'tpus', 'tpu-vm', 'ssh', args.tpu_name, '--zone', args.tpu_zone, '--command', args.command, '--worker', 'all']
    if args.debug:
        print(f'Running {' '.join(cmd)}')
        return
    subprocess.run(cmd)
    print('Successfully setup pod.')