import numpy as np
from ... import Point, PolyLineROI
from ... import functions as pgfn
from ... import metaarray as metaarray
from . import functions
from .common import CtrlNode, PlottingCtrlNode, metaArrayWrapper
class ButterworthNotch(CtrlNode):
    """Butterworth notch filter"""
    nodeName = 'ButterworthNotchFilter'
    uiTemplate = [('low_wPass', 'spin', {'value': 1000.0, 'step': 1, 'dec': True, 'bounds': [0.0, None], 'suffix': 'Hz', 'siPrefix': True}), ('low_wStop', 'spin', {'value': 2000.0, 'step': 1, 'dec': True, 'bounds': [0.0, None], 'suffix': 'Hz', 'siPrefix': True}), ('low_gPass', 'spin', {'value': 2.0, 'step': 1, 'dec': True, 'bounds': [0.0, None], 'suffix': 'dB', 'siPrefix': True}), ('low_gStop', 'spin', {'value': 20.0, 'step': 1, 'dec': True, 'bounds': [0.0, None], 'suffix': 'dB', 'siPrefix': True}), ('high_wPass', 'spin', {'value': 3000.0, 'step': 1, 'dec': True, 'bounds': [0.0, None], 'suffix': 'Hz', 'siPrefix': True}), ('high_wStop', 'spin', {'value': 4000.0, 'step': 1, 'dec': True, 'bounds': [0.0, None], 'suffix': 'Hz', 'siPrefix': True}), ('high_gPass', 'spin', {'value': 2.0, 'step': 1, 'dec': True, 'bounds': [0.0, None], 'suffix': 'dB', 'siPrefix': True}), ('high_gStop', 'spin', {'value': 20.0, 'step': 1, 'dec': True, 'bounds': [0.0, None], 'suffix': 'dB', 'siPrefix': True}), ('bidir', 'check', {'checked': True})]

    def processData(self, data):
        s = self.stateGroup.state()
        low = functions.butterworthFilter(data, bidir=s['bidir'], btype='low', wPass=s['low_wPass'], wStop=s['low_wStop'], gPass=s['low_gPass'], gStop=s['low_gStop'])
        high = functions.butterworthFilter(data, bidir=s['bidir'], btype='high', wPass=s['high_wPass'], wStop=s['high_wStop'], gPass=s['high_gPass'], gStop=s['high_gStop'])
        return low + high