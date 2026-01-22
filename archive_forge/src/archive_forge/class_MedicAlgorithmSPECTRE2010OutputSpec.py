import os
from ..base import (
class MedicAlgorithmSPECTRE2010OutputSpec(TraitedSpec):
    outOriginal = File(desc='If Output in Original Space Flag is true then outputs the original input volume. Otherwise outputs the axialy reoriented input volume.', exists=True)
    outStripped = File(desc='Skullstripped result of the input volume with just the brain.', exists=True)
    outMask = File(desc='Binary Mask of the skullstripped result with just the brain', exists=True)
    outPrior = File(desc='Probability prior from the atlas registrations', exists=True)
    outFANTASM = File(desc='Tissue classification of the whole input volume.', exists=True)
    outd0 = File(desc='Initial Brainmask', exists=True)
    outMidsagittal = File(desc='Plane dividing the brain hemispheres', exists=True)
    outSplitHalves = File(desc='Skullstripped mask of the brain with the hemispheres divided.', exists=True)
    outSegmentation = File(desc='2D image showing the tissue classification on the midsagittal plane', exists=True)