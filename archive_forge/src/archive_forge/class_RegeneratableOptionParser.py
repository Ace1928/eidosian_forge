import copy
import gyp.input
import argparse
import os.path
import re
import shlex
import sys
import traceback
from gyp.common import GypError
class RegeneratableOptionParser(argparse.ArgumentParser):

    def __init__(self, usage):
        self.__regeneratable_options = {}
        argparse.ArgumentParser.__init__(self, usage=usage)

    def add_argument(self, *args, **kw):
        """Add an option to the parser.

    This accepts the same arguments as ArgumentParser.add_argument, plus the
    following:
      regenerate: can be set to False to prevent this option from being included
                  in regeneration.
      env_name: name of environment variable that additional values for this
                option come from.
      type: adds type='path', to tell the regenerator that the values of
            this option need to be made relative to options.depth
    """
        env_name = kw.pop('env_name', None)
        if 'dest' in kw and kw.pop('regenerate', True):
            dest = kw['dest']
            type = kw.get('type')
            if type == 'path':
                kw['type'] = str
            self.__regeneratable_options[dest] = {'action': kw.get('action'), 'type': type, 'env_name': env_name, 'opt': args[0]}
        argparse.ArgumentParser.add_argument(self, *args, **kw)

    def parse_args(self, *args):
        values, args = argparse.ArgumentParser.parse_known_args(self, *args)
        values._regeneration_metadata = self.__regeneratable_options
        return (values, args)