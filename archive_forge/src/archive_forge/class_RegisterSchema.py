import argparse
import functools
import json
import os
import pathlib
from collections import defaultdict, namedtuple, OrderedDict
from dataclasses import dataclass, field
from typing import (
import yaml
import torchgen.api.dispatcher as dispatcher
import torchgen.api.meta as meta
import torchgen.api.native as native
import torchgen.api.structured as structured
import torchgen.dest as dest
from torchgen.api import cpp
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.context import (
from torchgen.gen_functionalization_type import (
from torchgen.gen_vmap_plumbing import gen_all_vmap_plumbing
from torchgen.model import (
from torchgen.native_function_generation import (
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import (
from torchgen.yaml_utils import YamlDumper, YamlLoader
@dataclass(frozen=True)
class RegisterSchema:
    selector: SelectiveBuilder
    known_tags: Dict[str, int] = field(default_factory=dict)

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> Optional[str]:
        if not self.selector.is_native_function_selected(f):
            return None
        tags = '{' + ', '.join((f'at::Tag::{tag}' for tag in sorted(f.tags))) + '}'
        if tags == '{}':
            return f'm.def({cpp_string(str(f.func))}, {{}});\n'
        maybe_tags = ''
        if tags not in self.known_tags:
            idx = len(self.known_tags)
            self.known_tags[tags] = idx
            maybe_tags = f'const std::vector<at::Tag> tags_{idx} = {tags};\n'
        return f'{maybe_tags}m.def({cpp_string(str(f.func))}, tags_{self.known_tags[tags]});\n'