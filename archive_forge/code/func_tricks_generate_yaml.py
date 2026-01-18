import os.path
import sys
import yaml
import time
import logging
from argh import arg, aliases, ArghParser, expects_obj
from wandb_watchdog.version import VERSION_STRING
from wandb_watchdog.utils import load_class
@aliases('generate-tricks-yaml')
@arg('trick_paths', nargs='*', help='Dotted paths for all the tricks you want to generate')
@arg('--python-path', default='.', help='paths separated by %s to add to the python path' % os.path.sep)
@arg('--append-to-file', default=None, help='appends the generated tricks YAML to a file; if not specified, prints to standard output')
@arg('-a', '--append-only', dest='append_only', default=False, help='if --append-to-file is not specified, produces output for appending instead of a complete tricks yaml file.')
@expects_obj
def tricks_generate_yaml(args):
    """
    Subcommand to generate Yaml configuration for tricks named on the command
    line.

    :param args:
        Command line argument options.
    """
    python_paths = path_split(args.python_path)
    add_to_sys_path(python_paths)
    output = StringIO()
    for trick_path in args.trick_paths:
        TrickClass = load_class(trick_path)
        output.write(TrickClass.generate_yaml())
    content = output.getvalue()
    output.close()
    header = yaml.dump({CONFIG_KEY_PYTHON_PATH: python_paths})
    header += '%s:\n' % CONFIG_KEY_TRICKS
    if args.append_to_file is None:
        if not args.append_only:
            content = header + content
        sys.stdout.write(content)
    else:
        if not os.path.exists(args.append_to_file):
            content = header + content
        output = open(args.append_to_file, 'ab')
        output.write(content)
        output.close()