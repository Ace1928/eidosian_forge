from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import cli_tree
from googlecloudsdk.calliope import markdown
from googlecloudsdk.calliope import walker
from googlecloudsdk.core import properties
from googlecloudsdk.core.document_renderers import render_document
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import pkg_resources
import six
def _UpdateTOC():
    """Updates the DevSIte TOC."""
    depth = len(command) - 1
    if not depth:
        return
    title = ' '.join(command)
    while depth >= len(self._need_section_tag):
        self._need_section_tag.append(False)
    if depth == 1:
        if is_group:
            if self._toc_main:
                self._toc_main.close()
            toc_path = os.path.join(directory, self._TOC)
            toc = files.FileWriter(toc_path)
            self._toc_main = toc
            toc.write('toc:\n')
            toc.write('- title: "%s"\n' % title)
            toc.write('  path: %s\n' % '/'.join([self._REFERENCE] + command[1:]))
            self._need_section_tag[depth] = True
        toc = self._toc_root
        indent = '  '
        if is_group:
            toc.write('%s- include: %s\n' % (indent, '/'.join([self._REFERENCE] + command[1:] + [self._TOC])))
            return
    else:
        toc = self._toc_main
        indent = '  ' * (depth - 1)
        if self._need_section_tag[depth - 1]:
            self._need_section_tag[depth - 1] = False
            toc.write('%ssection:\n' % indent)
        title = command[-1]
    toc.write('%s- title: "%s"\n' % (indent, title))
    toc.write('%s  path: %s\n' % (indent, '/'.join([self._REFERENCE] + command[1:])))
    self._need_section_tag[depth] = is_group