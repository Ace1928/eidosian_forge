from __future__ import division, print_function
import sys
import os
import io
import re
import glob
import math
import zlib
import time
import json
import enum
import struct
import pathlib
import warnings
import binascii
import tempfile
import datetime
import threading
import collections
import multiprocessing
import concurrent.futures
import numpy
def CZ_LSMINFO_READERS():
    return {'ScanInformation': read_lsm_scaninfo, 'TimeStamps': read_lsm_timestamps, 'EventList': read_lsm_eventlist, 'ChannelColors': read_lsm_channelcolors, 'Positions': read_lsm_floatpairs, 'TilePositions': read_lsm_floatpairs, 'VectorOverlay': None, 'InputLut': None, 'OutputLut': None, 'TimeIntervall': None, 'ChannelDataTypes': None, 'KsData': None, 'Roi': None, 'BleachRoi': None, 'NextRecording': None, 'MeanOfRoisOverlay': None, 'TopoIsolineOverlay': None, 'TopoProfileOverlay': None, 'ChannelWavelength': None, 'SphereCorrection': None, 'ChannelFactors': None, 'UnmixParameters': None, 'AcquisitionParameters': None, 'Characteristics': None}