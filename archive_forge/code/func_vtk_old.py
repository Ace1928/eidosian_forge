import os
from .. import logging
def vtk_old():
    """Checks if VTK uses the old-style pipeline (VTK<6.0)"""
    global _vtk_version
    if _vtk_version is None:
        raise RuntimeException('VTK is not correctly installed.')
    return _vtk_version[0] < 6