from .polynomial import Polynomial
from .ptolemyVarietyPrimeIdealGroebnerBasis import PtolemyVarietyPrimeIdealGroebnerBasis
from . import processFileBase
from . import utilities
import snappy
import re
import sys
import tempfile
import subprocess
import shutil
def triangulation_from_magma(text):
    """
    Reads the output from a magma computation and extracts the manifold for
    which this output contains solutions.
    """
    return processFileBase.get_manifold(text)