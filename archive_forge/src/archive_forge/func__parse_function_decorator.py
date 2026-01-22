import ast
import json
import os
import sys
import tokenize
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
def _parse_function_decorator(self, func_name: str, node: AstDecorator):
    """Parse function decorators."""
    if isinstance(node, ast.Attribute):
        name = node.value.id
        value = node.attr
        if value == 'setter':
            idx = self._index_of_property(name)
            if idx != -1:
                self._properties[idx]['write'] = func_name
        return
    if isinstance(node, ast.Call):
        name = _name(node.func)
        if name == 'Property':
            if node.args:
                type = _python_to_cpp_type(_name(node.args[0]))
                prop = self._create_property_entry(func_name, type, func_name)
                _parse_property_kwargs(node.keywords, prop)
                self._properties.append(prop)
        elif name == 'Slot':
            self._slots.append(_parse_slot(func_name, node))
        else:
            print('Unknown decorator with parameters:', name, file=sys.stderr)