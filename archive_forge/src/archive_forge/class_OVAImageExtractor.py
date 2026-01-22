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
class OVAImageExtractor(object):
    """Extracts and parses the uploaded OVA package

    A class that extracts the disk image and OVF file from an OVA
    tar archive. Parses the OVF file for metadata of interest.
    """

    def __init__(self):
        self.interested_properties = []
        self._load_interested_properties()

    def extract(self, ova):
        """Extracts disk image and OVF file from OVA package

        Extracts a single disk image and OVF from OVA tar archive and calls
        OVF parser method.

        :param ova: a file object containing the OVA file
        :returns: a tuple of extracted disk file object and dictionary of
            properties parsed from the OVF file
        :raises RuntimeError: an error for malformed OVA and OVF files
        """
        with tarfile.open(fileobj=ova) as tar_file:
            filenames = tar_file.getnames()
            ovf_filename = next((filename for filename in filenames if filename.endswith('.ovf')), None)
            if ovf_filename:
                ovf = tar_file.extractfile(ovf_filename)
                disk_name, properties = self._parse_OVF(ovf)
                ovf.close()
            else:
                raise RuntimeError(_('Could not find OVF file in OVA archive file.'))
            disk = tar_file.extractfile(disk_name)
            return (disk, properties)

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

    def _load_interested_properties(self):
        """Find the OVF properties config file and load it.

        OVF properties config file specifies which metadata of interest to
        extract. Reads in a JSON file named 'ovf-metadata.json' if available.
        See example file at etc/ovf-metadata.json.sample.
        """
        filename = 'ovf-metadata.json'
        match = CONF.find_file(filename)
        if match:
            with open(match, 'r') as properties_file:
                properties = json.loads(properties_file.read())
                self.interested_properties = properties.get('cim_pasd', [])
                if not self.interested_properties:
                    msg = _LW('OVF metadata of interest was not specified in ovf-metadata.json config file. Please set "cim_pasd" to a list of interested CIM_ProcessorAllocationSettingData properties.')
                    LOG.warning(msg)
        else:
            LOG.warning(_LW('OVF properties config file "ovf-metadata.json" was not found.'))