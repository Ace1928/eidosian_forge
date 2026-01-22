from xml.sax.saxutils import escape
import os.path
import subprocess
import gyp
import gyp.common
import gyp.msvs_emulation
import shlex
import xml.etree.cElementTree as ET
def GetJavaSourceDirs(target_list, target_dicts, toplevel_dir):
    """Generates a sequence of all likely java package root directories."""
    for target_name in target_list:
        target = target_dicts[target_name]
        for action in target.get('actions', []):
            for input_ in action['inputs']:
                if os.path.splitext(input_)[1] == '.java' and (not input_.startswith('$')):
                    dir_ = os.path.dirname(os.path.join(os.path.dirname(target_name), input_))
                    parent_search = dir_
                    while os.path.basename(parent_search) not in ['src', 'java']:
                        parent_search, _ = os.path.split(parent_search)
                        if not parent_search or parent_search == toplevel_dir:
                            yield dir_
                            break
                    else:
                        yield parent_search