from __future__ import (absolute_import, division, print_function)
import importlib
import os
import re
import sys
import textwrap
import yaml
def write_doc_fragment(name, doc_fragment):
    fn = doc_fragment_fn(name)
    data = doc_fragment.serialize_to_string()
    with open(fn, 'w', encoding='utf-8') as f:
        f.write(data)