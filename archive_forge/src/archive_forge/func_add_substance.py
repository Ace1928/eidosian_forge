import os
import subprocess
import shutil
import tempfile
def add_substance(key):
    fc = 'black'
    if key in categories['depleted']:
        fc = colors[0]
    if key in categories['accumulated']:
        fc = colors[1]
    label = ('$%s$' if tex else '%s') % getattr(rsys.substances[key], 'latex_name' if tex else 'name')
    lines.append(ind + '"{key}" [fontcolor={fc} label="{lbl}"];\n'.format(key=key, fc=fc, lbl=label))