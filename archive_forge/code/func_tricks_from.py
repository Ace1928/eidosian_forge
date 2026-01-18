import os.path
import sys
import yaml
import time
import logging
from argh import arg, aliases, ArghParser, expects_obj
from wandb_watchdog.version import VERSION_STRING
from wandb_watchdog.utils import load_class
@aliases('tricks')
@arg('files', nargs='*', help='perform tricks from given file')
@arg('--python-path', default='.', help='paths separated by %s to add to the python path' % os.path.sep)
@arg('--interval', '--timeout', dest='timeout', default=1.0, help='use this as the polling interval/blocking timeout')
@arg('--recursive', default=True, help='recursively monitor paths')
@expects_obj
def tricks_from(args):
    """
    Subcommand to execute tricks from a tricks configuration file.

    :param args:
        Command line argument options.
    """
    from watchdog.observers import Observer
    add_to_sys_path(path_split(args.python_path))
    observers = []
    for tricks_file in args.files:
        observer = Observer(timeout=args.timeout)
        if not os.path.exists(tricks_file):
            raise IOError('cannot find tricks file: %s' % tricks_file)
        config = load_config(tricks_file)
        try:
            tricks = config[CONFIG_KEY_TRICKS]
        except KeyError:
            raise KeyError("No `%s' key specified in %s." % (CONFIG_KEY_TRICKS, tricks_file))
        if CONFIG_KEY_PYTHON_PATH in config:
            add_to_sys_path(config[CONFIG_KEY_PYTHON_PATH])
        dir_path = os.path.dirname(tricks_file)
        if not dir_path:
            dir_path = os.path.relpath(os.getcwd())
        schedule_tricks(observer, tricks, dir_path, args.recursive)
        observer.start()
        observers.append(observer)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        for o in observers:
            o.unschedule_all()
            o.stop()
    for o in observers:
        o.join()