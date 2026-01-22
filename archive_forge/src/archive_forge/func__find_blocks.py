from ase import Atom, Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.utils import reader
import re
import xml.etree.ElementTree as ET
def _find_blocks(fp, tag, stopwords='[qbox]'):
    """Find and parse a certain block of the file.

    Reads a file sequentially and stops when it either encounters the
    end of the file, or until the it encounters a line that contains a
    user-defined string *after it has already found at least one
    desired block*. Use the stopwords ``[qbox]`` to read until the next
    command is issued.

    Groups the text between the first line that contains <tag> and the
    next line that contains </tag>, inclusively. The function then
    parses the XML and returns the Element object.

    Inputs:
        fp        - file-like object, file to be read from
        tag       - str, tag to search for (e.g., 'iteration').
                    `None` if you want to read until the end of the file
        stopwords - str, halt parsing if a line containing this string
                    is encountered

    Returns:
        list of xml.ElementTree, parsed XML blocks found by this class
    """
    start_tag = '<%s' % tag
    end_tag = '</%s>' % tag
    blocks = []
    cur_block = []
    in_block = False
    for line in fp:
        if start_tag in line:
            if in_block:
                raise Exception('Parsing failed: Encountered nested block')
            else:
                in_block = True
        if in_block:
            cur_block.append(line)
        if stopwords is not None:
            if stopwords in line and len(blocks) > 0:
                break
        if end_tag in line:
            if in_block:
                blocks.append(cur_block)
                cur_block = []
                in_block = False
            else:
                raise Exception('Parsing failed: End tag found before start tag')
    blocks = [''.join(b) for b in blocks]
    blocks = [re_find_bad_xml.sub('<\\1\\2_expectation_\\3', b) for b in blocks]
    return [ET.fromstring(b) for b in blocks]