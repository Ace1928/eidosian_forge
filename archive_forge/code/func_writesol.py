from .core import LpSolver_CMD, LpSolver, subprocess, PulpSolverError, clock, log
from .. import constants
import os
import warnings
def writesol(self, filename, vs):
    """Writes a CPLEX solution file"""
    try:
        import xml.etree.ElementTree as et
    except ImportError:
        import elementtree.ElementTree as et
    root = et.Element('CPLEXSolution', version='1.2')
    attrib_head = dict()
    attrib_quality = dict()
    et.SubElement(root, 'header', attrib=attrib_head)
    et.SubElement(root, 'header', attrib=attrib_quality)
    variables = et.SubElement(root, 'variables')
    values = [(v.name, v.value()) for v in vs if v.value() is not None]
    for index, (name, value) in enumerate(values):
        attrib_vars = dict(name=name, value=str(value), index=str(index))
        et.SubElement(variables, 'variable', attrib=attrib_vars)
    mst = et.ElementTree(root)
    mst.write(filename, encoding='utf-8', xml_declaration=True)
    return True