import os
import re
import sys
import unittest
from collections import defaultdict
from kivy.core.image import ImageLoader
def _test_image(self, fd, ctx, loadercls, imgdata):
    w, h, pixels, pitch = imgdata._data[0].get_mipmap(0)
    fmt = imgdata._data[0].fmt
    if not isinstance(pixels, bytes):
        pixels = bytearray(pixels)

    def debug():
        if not DEBUG:
            return
        print('    format: {}x{} {}'.format(w, h, fmt))
        print('     pitch: got {}, want {}'.format(pitch, want_pitch))
        print('      want: {} in {}'.format(fd['pattern'], fmt))
        print('       got: {}'.format(bytearray(pixels)))
    want_pitch = pitch == 0 and bytes_per_pixel(fmt) * w or pitch
    if pitch == 0 and bytes_per_pixel(fmt) * w * h != len(pixels):
        ctx.dbg('PITCH', 'pitch=0, expected fmt={} to be unaligned @ ({}bpp) = {} bytes, got {}'.format(fmt, bytes_per_pixel(fmt), bytes_per_pixel(fmt) * w * h, len(pixels)))
    elif pitch and want_pitch != pitch:
        ctx.dbg('PITCH', 'fmt={}, pitch={}, expected {}'.format(fmt, pitch, want_pitch))
    success, msgs = match_prediction(pixels, fmt, fd, pitch)
    if not success:
        if not msgs:
            ctx.fail('Unknown error')
        elif len(msgs) == 1:
            ctx.fail(msgs[0])
        else:
            for m in msgs:
                ctx.dbg('PREDICT', m)
            ctx.fail('{} errors, see debug output: {}'.format(len(msgs), msgs[-1]))
        debug()
    elif fd['require_alpha'] and (not has_alpha(fmt)):
        ctx.fail('Missing expected alpha channel')
        debug()
    elif fd['w'] != w:
        ctx.fail('Width mismatch, want {} got {}'.format(fd['w'], w))
        debug()
    elif fd['h'] != h:
        ctx.fail('Height mismatch, want {} got {}'.format(fd['h'], h))
        debug()
    elif w != 1 and h != 1:
        ctx.fail('v0 test protocol mandates w=1 or h=1')
        debug()
    else:
        ctx.ok('Passed test as {}x{} {}'.format(w, h, fmt))
    sys.stdout.flush()