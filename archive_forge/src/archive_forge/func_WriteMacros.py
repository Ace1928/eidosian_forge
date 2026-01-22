from xml.sax.saxutils import escape
import os.path
import subprocess
import gyp
import gyp.common
import gyp.msvs_emulation
import shlex
import xml.etree.cElementTree as ET
def WriteMacros(out, eclipse_langs, defines):
    """Write the macros section of a CDT settings export file."""
    out.write('  <section name="org.eclipse.cdt.internal.ui.wizards.settingswizards.Macros">\n')
    out.write('    <language name="holder for library settings"></language>\n')
    for lang in eclipse_langs:
        out.write('    <language name="%s">\n' % lang)
        for key in sorted(defines):
            out.write('      <macro><name>%s</name><value>%s</value></macro>\n' % (escape(key), escape(defines[key])))
        out.write('    </language>\n')
    out.write('  </section>\n')