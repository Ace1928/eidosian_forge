import numpy as np
from ... import Point, PolyLineROI
from ... import functions as pgfn
from ... import metaarray as metaarray
from . import functions
from .common import CtrlNode, PlottingCtrlNode, metaArrayWrapper
class Butterworth(CtrlNode):
    """Butterworth filter"""
    nodeName = 'ButterworthFilter'
    uiTemplate = [('band', 'combo', {'values': ['lowpass', 'highpass'], 'index': 0}), ('wPass', 'spin', {'value': 1000.0, 'step': 1, 'dec': True, 'bounds': [0.0, None], 'suffix': 'Hz', 'siPrefix': True}), ('wStop', 'spin', {'value': 2000.0, 'step': 1, 'dec': True, 'bounds': [0.0, None], 'suffix': 'Hz', 'siPrefix': True}), ('gPass', 'spin', {'value': 2.0, 'step': 1, 'dec': True, 'bounds': [0.0, None], 'suffix': 'dB', 'siPrefix': True}), ('gStop', 'spin', {'value': 20.0, 'step': 1, 'dec': True, 'bounds': [0.0, None], 'suffix': 'dB', 'siPrefix': True}), ('bidir', 'check', {'checked': True})]

    def processData(self, data):
        s = self.stateGroup.state()
        if s['band'] == 'lowpass':
            mode = 'low'
        else:
            mode = 'high'
        ret = functions.butterworthFilter(data, bidir=s['bidir'], btype=mode, wPass=s['wPass'], wStop=s['wStop'], gPass=s['gPass'], gStop=s['gStop'])
        return ret