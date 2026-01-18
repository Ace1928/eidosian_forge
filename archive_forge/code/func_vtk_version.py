import os
from .. import logging
def vtk_version():
    """Get VTK version"""
    global _vtk_version
    return _vtk_version