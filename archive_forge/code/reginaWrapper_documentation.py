from regina import NTriangulation, writeXMLFile, readXMLFile
import tempfile
import os
from . import manifoldMethods
from . import utilities

            Given a tetrahedron, return for each face which inbound
            or outbound generator it belongs to.
            