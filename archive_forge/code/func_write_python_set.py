import json
import xml.etree.ElementTree as ET
from langcodes.util import data_filename
from langcodes.registry_parser import parse_registry
def write_python_set(outfile, name, s):
    print(f'{name} = {{', file=outfile)
    for key in sorted(set(s)):
        print(f'    {key!r},', file=outfile)
    print('}', file=outfile)