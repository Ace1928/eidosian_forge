import os
import time
import warnings
import numpy as np
from .. import _HAS_FFMPEG
from ..utils import *
def _warmStart(self, M, N, C, dtype):
    self.warmStarted = True
    if '-pix_fmt' not in self.inputdict:
        if dtype.kind == 'u' and dtype.itemsize == 2:
            suffix = 'le' if dtype.byteorder else 'be'
            if C == 1:
                if self.NEED_RGB2GRAY_HACK:
                    self.inputdict['-pix_fmt'] = 'rgb48' + suffix
                    self.rgb2grayhack = True
                    C = 3
                else:
                    self.inputdict['-pix_fmt'] = 'gray16' + suffix
            elif C == 2:
                self.inputdict['-pix_fmt'] = 'ya16' + suffix
            elif C == 3:
                self.inputdict['-pix_fmt'] = 'rgb48' + suffix
            elif C == 4:
                self.inputdict['-pix_fmt'] = 'rgba64' + suffix
            else:
                raise NotImplemented
        elif C == 1:
            if self.NEED_RGB2GRAY_HACK:
                self.inputdict['-pix_fmt'] = 'rgb24'
                self.rgb2grayhack = True
                C = 3
            else:
                self.inputdict['-pix_fmt'] = 'gray'
        elif C == 2:
            self.inputdict['-pix_fmt'] = 'ya8'
        elif C == 3:
            self.inputdict['-pix_fmt'] = 'rgb24'
        elif C == 4:
            self.inputdict['-pix_fmt'] = 'rgba'
        else:
            raise NotImplemented
    self.bpp = bpplut[self.inputdict['-pix_fmt']][1]
    self.inputNumChannels = bpplut[self.inputdict['-pix_fmt']][0]
    bitpercomponent = self.bpp // self.inputNumChannels
    if bitpercomponent == 8:
        self.dtype = np.dtype('u1')
    elif bitpercomponent == 16:
        suffix = self.inputdict['-pix_fmt'][-2:]
        if suffix == 'le':
            self.dtype = np.dtype('<u2')
        elif suffix == 'be':
            self.dtype = np.dtype('>u2')
    else:
        raise ValueError(self.inputdict['-pix_fmt'] + 'is not a valid pix_fmt for numpy conversion')
    assert self.inputNumChannels == C, 'Failed to pass the correct number of channels %d for the pixel format %s.' % (self.inputNumChannels, self.inputdict['-pix_fmt'])
    if '-s' in self.inputdict:
        widthheight = self.inputdict['-s'].split('x')
        self.inputwidth = np.int(widthheight[0])
        self.inputheight = np.int(widthheight[1])
    else:
        self.inputdict['-s'] = str(N) + 'x' + str(M)
        self.inputwidth = N
        self.inputheight = M
    if self.extension == '.yuv':
        if '-pix_fmt' not in self.outputdict:
            self.outputdict['-pix_fmt'] = self.DEFAULT_OUTPUT_PIX_FMT
            if self.verbosity > 0:
                warnings.warn('No output color space provided. Assuming {}.'.format(self.DEFAULT_OUTPUT_PIX_FMT), UserWarning)
    self._createProcess(self.inputdict, self.outputdict, self.verbosity)