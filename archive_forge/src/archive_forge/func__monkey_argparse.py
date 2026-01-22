import argparse
import copy
import json
import os
import re
import sys
import yaml
import wandb
from wandb import trigger
from wandb.util import add_import_hook, get_optional_module
def _monkey_argparse():
    argparse._ArgumentParser = argparse.ArgumentParser

    def _install():
        argparse.ArgumentParser = MonitoredArgumentParser

    def _uninstall():
        argparse.ArgumentParser = argparse._ArgumentParser

    def monitored(self, args, unknown=None):
        global _args_argparse
        _args_argparse = copy.deepcopy(vars(args))

    class MonitoredArgumentParser(argparse._ArgumentParser):

        def __init__(self, *args, **kwargs):
            _uninstall()
            super().__init__(*args, **kwargs)
            _install()

        def parse_args(self, *args, **kwargs):
            args = super().parse_args(*args, **kwargs)
            return args

        def parse_known_args(self, *args, **kwargs):
            args, unknown = super().parse_known_args(*args, **kwargs)
            if self._callback:
                self._callback(args, unknown=unknown)
            return (args, unknown)
    _install()
    argparse.ArgumentParser._callback = monitored