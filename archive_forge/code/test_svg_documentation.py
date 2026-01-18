import os
import unittest
from xml.dom.minidom import parseString as parse_xml_string
from shapely.geometry import (
from shapely.geometry.collection import GeometryCollection
Helper function to check XML and debug SVG