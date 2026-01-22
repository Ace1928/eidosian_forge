from __future__ import annotations
import sys
from enum import IntEnum
from . import Image
class ImageCmsTransform(Image.ImagePointHandler):
    """
    Transform.  This can be used with the procedural API, or with the standard
    :py:func:`~PIL.Image.Image.point` method.

    Will return the output profile in the ``output.info['icc_profile']``.
    """

    def __init__(self, input, output, input_mode, output_mode, intent=Intent.PERCEPTUAL, proof=None, proof_intent=Intent.ABSOLUTE_COLORIMETRIC, flags=0):
        if proof is None:
            self.transform = core.buildTransform(input.profile, output.profile, input_mode, output_mode, intent, flags)
        else:
            self.transform = core.buildProofTransform(input.profile, output.profile, proof.profile, input_mode, output_mode, intent, proof_intent, flags)
        self.input_mode = self.inputMode = input_mode
        self.output_mode = self.outputMode = output_mode
        self.output_profile = output

    def point(self, im):
        return self.apply(im)

    def apply(self, im, imOut=None):
        im.load()
        if imOut is None:
            imOut = Image.new(self.output_mode, im.size, None)
        self.transform.apply(im.im.id, imOut.im.id)
        imOut.info['icc_profile'] = self.output_profile.tobytes()
        return imOut

    def apply_in_place(self, im):
        im.load()
        if im.mode != self.output_mode:
            msg = 'mode mismatch'
            raise ValueError(msg)
        self.transform.apply(im.im.id, im.im.id)
        im.info['icc_profile'] = self.output_profile.tobytes()
        return im