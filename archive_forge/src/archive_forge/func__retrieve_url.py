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
def _retrieve_url(url):
    overview_url = 'http://ptolemy.unhyperbolic.org/data/overview.html'
    try:
        sigalrm_handler = None
        if hasattr(signal, 'SIGALRM'):
            sigalrm_handler = signal.signal(signal.SIGALRM, signal.SIG_IGN)
        r = Request(url, headers={'User-Agent': 'Wget/1.20.3'})
        s = urlopen(r)
    except HTTPError as e:
        if e.code in [404, 406]:
            raise PtolemyFileMissingError('The ptolemy variety has probably not been computed yet, see %s (%s)' % (overview_url, e))
        else:
            raise PtolemyFileMissingError('The ptolemy variety has probably not been computed yet or the given data_url or environment variable PTOLEMY_DATA_URL is not configured correctly: %s. Also see %s' % (e, overview_url))
    except IOError as e:
        if url[:5] == 'http:':
            raise RuntimeError('Problem connecting to server while retrieving %s: %s' % (url, e))
        else:
            raise PtolemyFileMissingError('The ptolemy variety has probably not been computed yet or the given data_url or environment variable PTOLEMY_DATA_URL is not configured correctly: %s. Also see %s' % (e, overview_url))
    finally:
        if sigalrm_handler:
            signal.signal(signal.SIGALRM, sigalrm_handler)
    text = s.read().decode('ascii').replace('\r\n', '\n')
    if url[:5] != 'http:':
        return text
    code = s.getcode()
    if code == 200:
        return text
    httpErr = '(HTTP Error %d while accessing %s)' % (code, url)
    if code in [404, 406]:
        raise PtolemyFileMissingError('The ptolemy variety has probably not been computed yet, see %s (%s)' % (overview_url, httpErr))
    raise RuntimeError('Problem retrieving file from server, please report to enischte@gmail.com: %s' % httpErr)