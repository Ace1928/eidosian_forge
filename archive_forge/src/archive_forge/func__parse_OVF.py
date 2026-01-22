import os
import re
import shutil
import tarfile
import urllib
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils as json
from taskflow.patterns import linear_flow as lf
from taskflow import task
from glance.i18n import _, _LW
def _parse_OVF(self, ovf):
    """Parses the OVF file

        Parses the OVF file for specified metadata properties. Interested
        properties must be specified in ovf-metadata.json conf file.

        The OVF file's qualified namespaces are removed from the included
        properties.

        :param ovf: a file object containing the OVF file
        :returns: a tuple of disk filename and a properties dictionary
        :raises RuntimeError: an error for malformed OVF file
        """

    def _get_namespace_and_tag(tag):
        """Separate and return the namespace and tag elements.

            There is no native support for this operation in elementtree
            package. See http://bugs.python.org/issue18304 for details.
            """
        m = re.match('\\{(.+)\\}(.+)', tag)
        if m:
            return (m.group(1), m.group(2))
        else:
            return ('', tag)
    disk_filename, file_elements, file_ref = (None, None, None)
    properties = {}
    for event, elem in ET.iterparse(ovf):
        if event == 'end':
            ns, tag = _get_namespace_and_tag(elem.tag)
            if ns in CIM_NS and tag in self.interested_properties:
                properties[CIM_NS[ns] + '_' + tag] = elem.text.strip() if elem.text else ''
            if tag == 'DiskSection':
                disks = [child for child in list(elem) if _get_namespace_and_tag(child.tag)[1] == 'Disk']
                if len(disks) > 1:
                    '\n                        Currently only single disk image extraction is\n                        supported.\n                        FIXME(dramakri): Support multiple images in OVA package\n                        '
                    raise RuntimeError(_('Currently, OVA packages containing multiple disk are not supported.'))
                disk = next(iter(disks))
                file_ref = next((value for key, value in disk.items() if _get_namespace_and_tag(key)[1] == 'fileRef'))
            if tag == 'References':
                file_elements = list(elem)
            if tag != 'File' and tag != 'Disk':
                elem.clear()
    for file_element in file_elements:
        file_id = next((value for key, value in file_element.items() if _get_namespace_and_tag(key)[1] == 'id'))
        if file_id != file_ref:
            continue
        disk_filename = next((value for key, value in file_element.items() if _get_namespace_and_tag(key)[1] == 'href'))
    return (disk_filename, properties)