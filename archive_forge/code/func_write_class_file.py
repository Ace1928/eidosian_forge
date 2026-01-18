import os
import sys
import shutil
import importlib
import textwrap
import re
import warnings
from ._all_keywords import r_keywords
from ._py_components_generation import reorder_props
def write_class_file(name, props, description, project_shortname, prefix=None, rpkg_data=None):
    props = reorder_props(props=props)
    write_help_file(name, props, description, prefix, rpkg_data)
    import_string = '# AUTO GENERATED FILE - DO NOT EDIT\n\n'
    class_string = generate_class_string(name, props, project_shortname, prefix)
    file_name = format_fn_name(prefix, name) + '.R'
    file_path = os.path.join('R', file_name)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(import_string)
        f.write(class_string)
    print('Generated {}'.format(file_name))