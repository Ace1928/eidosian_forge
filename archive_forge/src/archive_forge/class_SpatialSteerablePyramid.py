import numpy as np
import scipy.misc as sc
import scipy.signal
import scipy.io
class SpatialSteerablePyramid:

    def __init__(self, height=4):
        """
    height is the total height, including highpass and lowpass
    """
        self.height = height

    def corr(self, A, fw):
        h, w = A.shape
        sy2 = np.int(np.floor((fw.shape[0] - 1) / 2))
        sx2 = np.int(np.floor((fw.shape[1] - 1) / 2))
        newpad = np.vstack((A[1:fw.shape[0] - sy2, :][::-1], A, A[h - (fw.shape[0] - sy2):h - 1, :][::-1]))
        newpad = np.hstack((newpad[:, 1:fw.shape[1] - sx2][:, ::-1], newpad, newpad[:, w - (fw.shape[1] - sx2):w - 1][:, ::-1]))
        newpad = newpad.astype(np.float32)
        return scipy.signal.correlate2d(newpad, fw, mode='valid').astype(np.float32)

    def buildLevs(self, lo0, lofilt, bfilts, edges, mHeight):
        if mHeight <= 0:
            return [lo0]
        bands = []
        for i in range(bfilts.shape[0]):
            filt = bfilts[i]
            bands.append(self.corr(lo0, filt))
        lo = self.corr(lo0, lofilt)[::2, ::2]
        bands = [bands] + self.buildLevs(lo, lofilt, bfilts, edges, mHeight - 1)
        return bands

    def decompose(self, inputimage, filtfile='sp1Filters', edges='symm'):
        inputimage = inputimage.astype(np.float32)
        if filtfile == 'sp5Filters':
            lo0filt, hi0filt, lofilt, bfilts, mtx, harmonics = load_sp5filters()
        else:
            raise (NotImplementedError, 'That filter configuration is not implemnted')
        h, w = inputimage.shape
        hi0 = self.corr(inputimage, hi0filt)
        lo0 = self.corr(inputimage, lo0filt)
        pyr = self.buildLevs(lo0, lofilt, bfilts, edges, self.height)
        pyr = [hi0] + pyr
        return pyr

    def extractSingleBand(self, inputimage, filtfile='sp1Filters', edges='symm', band=0, level=1):
        inputimage = inputimage.astype(np.float32)
        if filtfile == 'sp5Filters':
            lo0filt, hi0filt, lofilt, bfilts, mtx, harmonics = load_sp5filters()
        else:
            raise (NotImplementedError, 'That filter configuration is not implemnted')
        h, w = inputimage.shape
        if level == 0:
            hi0 = self.corr(inputimage, hi0filt)
            singleband = hi0
        else:
            lo0 = self.corr(inputimage, lo0filt)
            for i in range(1, level):
                lo0 = self.corr(lo0, lofilt)[::2, ::2]
            filt = bfilts[band]
            singleband = self.corr(lo0, filt)
        return singleband