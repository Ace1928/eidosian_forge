import os
import re
import sys
import unittest
from collections import defaultdict
from kivy.core.image import ImageLoader
@unittest.skipIf(not os.path.isdir(asset(ASSETDIR)), "Need 'make image-testsuite' to run test")
class ImageLoaderTestCase(unittest.TestCase):

    def setUp(self):
        self._context = None
        self._prepare_images()

    def tearDown(self):
        if not DEBUG or not self._context:
            return
        ctx = self._context
        il = ctx.loadercls.__name__
        stats = ctx.stats
        keys = set([k for x in stats.values() for k in x.keys()])
        sg = stats.get
        for k in sorted(keys):
            ok, skip, fail = (sg('ok', {}), sg('skip', {}), sg('fail', {}))
            print('REPORT {} {}: ok={}, skip={}, fail={}'.format(il, k, ok.get(k, 0), skip.get(k, 0), fail.get(k, 0)))

    def _prepare_images(self):
        if hasattr(self, '_image_files'):
            return
        self._image_files = {}
        for filename in os.listdir(asset(ASSETDIR)):
            matches = v0_FILE_RE.match(filename)
            if not matches:
                continue
            w, h, pat, alpha, fmtinfo, tst, encoder, ext = matches.groups()
            self._image_files[filename] = {'filename': filename, 'w': int(w), 'h': int(h), 'pattern': pat, 'alpha': alpha, 'fmtinfo': fmtinfo, 'testname': tst, 'encoder': encoder, 'ext': ext, 'require_alpha': 'BINARY' in tst or 'ALPHA' in tst}

    def _test_imageloader(self, loadercls, extensions=None):
        if not loadercls:
            return
        if not extensions:
            extensions = loadercls.extensions()
        ctx = _TestContext(loadercls)
        self._context = ctx
        for filename in sorted(self._image_files.keys()):
            filedata = self._image_files[filename]
            if filedata['ext'] not in extensions:
                continue
            try:
                ctx.start(filename, filedata)
                result = loadercls(asset(ASSETDIR, filename), keep_data=True)
                if not result:
                    raise Exception('invalid result')
            except:
                ctx.skip('Error loading file, result=None')
                continue
            self._test_image(filedata, ctx, loadercls, result)
            ctx.end()
        ok, skip, fail, stats = ctx.results
        if fail:
            self.fail('{}: {} passed, {} skipped, {} failed'.format(loadercls.__name__, ok, skip, fail))
        return ctx

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

    def test_ImageLoaderSDL2(self):
        loadercls = LOADERS.get('ImageLoaderSDL2')
        if loadercls:
            exts = list(loadercls.extensions()) + ['gif']
            ctx = self._test_imageloader(loadercls, exts)

    def test_ImageLoaderPIL(self):
        loadercls = LOADERS.get('ImageLoaderPIL')
        ctx = self._test_imageloader(loadercls)

    def test_ImageLoaderPygame(self):
        loadercls = LOADERS.get('ImageLoaderPygame')
        ctx = self._test_imageloader(loadercls)

    def test_ImageLoaderFFPy(self):
        loadercls = LOADERS.get('ImageLoaderFFPy')
        ctx = self._test_imageloader(loadercls)

    def test_ImageLoaderGIF(self):
        loadercls = LOADERS.get('ImageLoaderGIF')
        ctx = self._test_imageloader(loadercls)

    def test_ImageLoaderDDS(self):
        loadercls = LOADERS.get('ImageLoaderDDS')
        ctx = self._test_imageloader(loadercls)

    def test_ImageLoaderTex(self):
        loadercls = LOADERS.get('ImageLoaderTex')
        ctx = self._test_imageloader(loadercls)

    def test_ImageLoaderImageIO(self):
        loadercls = LOADERS.get('ImageLoaderImageIO')
        ctx = self._test_imageloader(loadercls)

    def test_missing_tests(self):
        for loader in ImageLoader.loaders:
            key = 'test_{}'.format(loader.__name__)
            msg = 'Missing ImageLoader test case: {}'.format(key)
            self.assertTrue(hasattr(self, key), msg)
            self.assertTrue(callable(getattr(self, key)), msg)