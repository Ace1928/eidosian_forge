from builtins import zip
from builtins import str
import os
import os.path as op
import sys
from xml.etree import cElementTree as ET
import pyxnat
def write_xml(xml_str, file_path, clean_tags=True):
    """Writing XML."""
    root = ET.fromstring(xml_str)
    if clean_tags:
        if 'ID' in root.attrib:
            del root.attrib['ID']
        tag = '{http://nrg.wustl.edu/xnat}sharing'
        for child in root.findall(tag):
            root.remove(child)
        for child in root.findall('{http://nrg.wustl.edu/xnat}out'):
            root.remove(child)
            break
        tag = '{http://nrg.wustl.edu/xnat}imageSession_ID'
        for child in root.findall(tag):
            root.remove(child)
        for child in root.findall('{http://nrg.wustl.edu/xnat}subject_ID'):
            root.remove(child)
        tag = '{http://nrg.wustl.edu/xnat}image_session_ID'
        for child in root.findall(tag):
            root.remove(child)
        for child in root.findall('{http://nrg.wustl.edu/xnat}scans'):
            root.remove(child)
            break
        for child in root.findall('{http://nrg.wustl.edu/xnat}assessors'):
            root.remove(child)
            break
        for child in root.findall('{http://nrg.wustl.edu/xnat}resources'):
            root.remove(child)
            break
        for child in root.findall('{http://nrg.wustl.edu/xnat}experiments'):
            root.remove(child)
            break
    try:
        ET.register_namespace('xnat', 'http://nrg.wustl.edu/xnat')
        ET.register_namespace('proc', 'http://nrg.wustl.edu/proc')
        ET.register_namespace('prov', 'http://www.nbirn.net/prov')
        ET.register_namespace('fs', 'http://nrg.wustl.edu/fs')
        ET.ElementTree(root).write(file_path)
    except IOError as error:
        print('ERROR:writing xml file: {}: {}'.format(file_path, str(error)))