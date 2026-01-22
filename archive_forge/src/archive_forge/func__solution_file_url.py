from . import matrix
from . import homology
from .polynomial import Polynomial
from .ptolemyObstructionClass import PtolemyObstructionClass
from .ptolemyGeneralizedObstructionClass import PtolemyGeneralizedObstructionClass
from .ptolemyVarietyPrimeIdealGroebnerBasis import PtolemyVarietyPrimeIdealGroebnerBasis
from . import processFileBase, processFileDispatch, processMagmaFile
from . import utilities
from string import Template
import signal
import re
import os
import sys
from urllib.request import Request, urlopen
from urllib.request import quote as urlquote
from urllib.error import HTTPError
def _solution_file_url(self, data_url=None, rur=False):
    if data_url is None:
        from . import DATA_URL as data_url
    if '://' not in data_url:
        if not data_url[0] == '/':
            data_url = '/' + data_url
        data_url = 'file://' + data_url
    if not data_url[-1] == '/':
        data_url = data_url + '/'
    if rur:
        ext = '.rur'
    else:
        ext = '.magma_out'
    filename = self.filename_base() + ext
    return data_url + self.path_to_file() + '/' + urlquote(filename)