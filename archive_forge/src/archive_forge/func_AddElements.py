from xml.sax.saxutils import escape
import os.path
import subprocess
import gyp
import gyp.common
import gyp.msvs_emulation
import shlex
import xml.etree.cElementTree as ET
def AddElements(kind, paths):
    rel_paths = set()
    for path in paths:
        if os.path.isabs(path):
            rel_paths.add(os.path.relpath(path, toplevel_dir))
        else:
            rel_paths.add(path)
    for path in sorted(rel_paths):
        entry_element = ET.SubElement(result, 'classpathentry')
        entry_element.set('kind', kind)
        entry_element.set('path', path)