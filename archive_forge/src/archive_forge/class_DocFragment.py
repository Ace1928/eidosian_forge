from __future__ import (absolute_import, division, print_function)
import importlib
import os
import re
import sys
import textwrap
import yaml
class DocFragment:

    def __init__(self, path, prefix_lines, name, lines):
        self.prefix_lines = prefix_lines
        self.name = name
        self.lines = lines
        try:
            self.data = yaml.safe_load('\n'.join(self.lines))
        except Exception as e:
            raise DocFragmentParseError(path, 'Error while parsing part {0}: {1}'.format(name, e))

    def recreate_lines(self):
        data = yaml.dump(self.data, default_flow_style=False, indent=4, Dumper=Dumper, sort_keys=False)
        self.lines = data.splitlines()

    def serialize_lines(self):
        return self.prefix_lines + ["    {0} = r'''".format(self.name)] + self.lines + ["'''"]