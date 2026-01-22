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
def PILATUS_HEADER():
    return {'Detector': ([slice(1, None)], str), 'Pixel_size': ([1, 4], float), 'Silicon': ([3], float), 'Exposure_time': ([1], float), 'Exposure_period': ([1], float), 'Tau': ([1], float), 'Count_cutoff': ([1], int), 'Threshold_setting': ([1], float), 'Gain_setting': ([1, 2], str), 'N_excluded_pixels': ([1], int), 'Excluded_pixels': ([1], str), 'Flat_field': ([1], str), 'Trim_file': ([1], str), 'Image_path': ([1], str), 'Wavelength': ([1], float), 'Energy_range': ([1, 2], float), 'Detector_distance': ([1], float), 'Detector_Voffset': ([1], float), 'Beam_xy': ([1, 2], float), 'Flux': ([1], str), 'Filter_transmission': ([1], float), 'Start_angle': ([1], float), 'Angle_increment': ([1], float), 'Detector_2theta': ([1], float), 'Polarization': ([1], float), 'Alpha': ([1], float), 'Kappa': ([1], float), 'Phi': ([1], float), 'Phi_increment': ([1], float), 'Chi': ([1], float), 'Chi_increment': ([1], float), 'Oscillation_axis': ([slice(1, None)], str), 'N_oscillations': ([1], int), 'Start_position': ([1], float), 'Position_increment': ([1], float), 'Shutter_time': ([1], float), 'Omega': ([1], float), 'Omega_increment': ([1], float)}