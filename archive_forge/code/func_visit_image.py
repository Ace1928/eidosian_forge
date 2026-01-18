import sys
import os
import os.path
import tempfile
import zipfile
from xml.dom import minidom
import time
import re
import copy
import itertools
import docutils
from docutils import frontend, nodes, utils, writers, languages
from docutils.readers import standalone
from docutils.transforms import references
def visit_image(self, node):
    if 'uri' in node.attributes:
        source = node.attributes['uri']
        if not (source.startswith('http:') or source.startswith('https:')):
            if not source.startswith(os.sep):
                docsource, line = utils.get_source_line(node)
                if docsource:
                    dirname = os.path.dirname(docsource)
                    if dirname:
                        source = '%s%s%s' % (dirname, os.sep, source)
            if not self.check_file_exists(source):
                self.document.reporter.warning('Cannot find image file %s.' % (source,))
                return
    else:
        return
    if source in self.image_dict:
        filename, destination = self.image_dict[source]
    else:
        self.image_count += 1
        filename = os.path.split(source)[1]
        destination = 'Pictures/1%08x%s' % (self.image_count, filename)
        if source.startswith('http:') or source.startswith('https:'):
            try:
                imgfile = urlopen(source)
                content = imgfile.read()
                imgfile.close()
                imgfile2 = tempfile.NamedTemporaryFile('wb', delete=False)
                imgfile2.write(content)
                imgfile2.close()
                imgfilename = imgfile2.name
                source = imgfilename
            except HTTPError:
                self.document.reporter.warning("Can't open image url %s." % (source,))
            spec = (source, destination)
        else:
            spec = (os.path.abspath(source), destination)
        self.embedded_file_list.append(spec)
        self.image_dict[source] = (source, destination)
    if self.in_paragraph:
        el1 = self.current_element
    else:
        el1 = SubElement(self.current_element, 'text:p', attrib={'text:style-name': self.rststyle('textbody')})
    el2 = el1
    if isinstance(node.parent, docutils.nodes.figure):
        el3, el4, el5, caption = self.generate_figure(node, source, destination, el2)
        attrib = {}
        el6, width = self.generate_image(node, source, destination, el5, attrib)
        if caption is not None:
            el6.tail = caption
    else:
        self.generate_image(node, source, destination, el2)