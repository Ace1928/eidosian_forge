from __future__ import (absolute_import, division, print_function)
import importlib
import os
import re
import sys
import textwrap
import yaml
class DocFragmentFile:

    def __init__(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
        self.prefix = []
        self.fragments = []
        self.fragments_by_name = {}
        where = 'prefix'
        for line in lines:
            if where == 'prefix':
                self.prefix.append(line)
                if line == 'class ModuleDocFragment(object):':
                    where = 'body'
                    body_prefix = []
                    body_name = None
                    body_lines = []
            elif where == 'body':
                if body_name is None:
                    m = DOC_FRAGMENT_START_MATCHER.match(line)
                    if m:
                        body_name = m.group(1)
                    else:
                        body_prefix.append(line)
                elif line == "'''":
                    fragment = DocFragment(path, body_prefix, body_name, body_lines)
                    self.fragments.append(fragment)
                    self.fragments_by_name[body_name] = fragment
                    body_prefix = []
                    body_name = None
                    body_lines = []
                else:
                    body_lines.append(line)
        if where == 'prefix':
            raise DocFragmentParseError(path, 'Cannot find body')

    def serialize_to_string(self):
        lines = []
        lines.extend(self.prefix)
        for fragment in self.fragments:
            lines.extend(fragment.serialize_lines())
        lines.append('')
        return '\n'.join(lines)