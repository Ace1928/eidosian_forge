from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import itertools
import logging
import os
import sys
from xml.dom import minidom
from absl.flags import _exceptions
from absl.flags import _flag
from absl.flags import _helpers
from absl.flags import _validators_classes
import six
def write_help_in_xml_format(self, outfile=None):
    """Outputs flag documentation in XML format.

    NOTE: We use element names that are consistent with those used by
    the C++ command-line flag library, from
    https://github.com/gflags/gflags.
    We also use a few new elements (e.g., <key>), but we do not
    interfere / overlap with existing XML elements used by the C++
    library.  Please maintain this consistency.

    Args:
      outfile: File object we write to.  Default None means sys.stdout.
    """
    doc = minidom.Document()
    all_flag = doc.createElement('AllFlags')
    doc.appendChild(all_flag)
    all_flag.appendChild(_helpers.create_xml_dom_element(doc, 'program', os.path.basename(sys.argv[0])))
    usage_doc = sys.modules['__main__'].__doc__
    if not usage_doc:
        usage_doc = '\nUSAGE: %s [flags]\n' % sys.argv[0]
    else:
        usage_doc = usage_doc.replace('%s', sys.argv[0])
    all_flag.appendChild(_helpers.create_xml_dom_element(doc, 'usage', usage_doc))
    key_flags = self.get_key_flags_for_module(sys.argv[0])
    flags_by_module = self.flags_by_module_dict()
    all_module_names = list(flags_by_module.keys())
    all_module_names.sort()
    for module_name in all_module_names:
        flag_list = [(f.name, f) for f in flags_by_module[module_name]]
        flag_list.sort()
        for unused_flag_name, flag in flag_list:
            is_key = flag in key_flags
            all_flag.appendChild(flag._create_xml_dom_element(doc, module_name, is_key=is_key))
    outfile = outfile or sys.stdout
    if six.PY2:
        outfile.write(doc.toprettyxml(indent='  ', encoding='utf-8'))
    else:
        outfile.write(doc.toprettyxml(indent='  ', encoding='utf-8').decode('utf-8'))
    outfile.flush()