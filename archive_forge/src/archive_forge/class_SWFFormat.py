import os
import zlib
import logging
from io import BytesIO
import numpy as np
from ..core import Format, read_n_bytes, image_as_uint
class SWFFormat(Format):
    """See :mod:`imageio.plugins.swf`"""

    def _can_read(self, request):
        tmp = request.firstbytes[0:3].decode('ascii', 'ignore')
        if tmp in ('FWS', 'CWS'):
            return True

    def _can_write(self, request):
        if request.extension in self.extensions:
            return True

    class Reader(Format.Reader):

        def _open(self, loop=False):
            if not _swf:
                load_lib()
            self._arg_loop = bool(loop)
            self._fp = self.request.get_file()
            tmp = self.request.firstbytes[0:3].decode('ascii', 'ignore')
            if tmp == 'FWS':
                pass
            elif tmp == 'CWS':
                bb = self._fp.read()
                bb = bb[:8] + zlib.decompress(bb[8:])
                self._fp = BytesIO(bb)
            else:
                raise IOError('This does not look like a valid SWF file')
            try:
                self._fp.seek(8)
                self._streaming_mode = False
            except Exception:
                self._streaming_mode = True
                self._fp_read(8)
            nbits = _swf.bits2int(self._fp_read(1), 5)
            nbits = 5 + nbits * 4
            Lrect = nbits / 8.0
            if Lrect % 1:
                Lrect += 1
            Lrect = int(Lrect)
            self._fp_read(Lrect + 3)
            self._imlocs = []
            if not self._streaming_mode:
                try:
                    while True:
                        isimage, sze, T, L1 = self._read_one_tag()
                        loc = self._fp.tell()
                        if isimage:
                            format = ord(self._fp_read(3)[2:])
                            if format == 5:
                                self._imlocs.append((loc, sze, T, L1))
                        self._fp.seek(loc + sze)
                except IndexError:
                    pass

        def _fp_read(self, n):
            return read_n_bytes(self._fp, n)

        def _close(self):
            pass

        def _get_length(self):
            if self._streaming_mode:
                return np.inf
            else:
                return len(self._imlocs)

        def _get_data(self, index):
            if index < 0:
                raise IndexError('Index in swf file must be > 0')
            if not self._streaming_mode:
                if self._arg_loop and self._imlocs:
                    index = index % len(self._imlocs)
                if index >= len(self._imlocs):
                    raise IndexError('Index out of bounds')
            if self._streaming_mode:
                while True:
                    isimage, sze, T, L1 = self._read_one_tag()
                    bb = self._fp_read(sze)
                    if isimage:
                        im = _swf.read_pixels(bb, 0, T, L1)
                        if im is not None:
                            return (im, {})
            else:
                loc, sze, T, L1 = self._imlocs[index]
                self._fp.seek(loc)
                bb = self._fp_read(sze)
                im = _swf.read_pixels(bb, 0, T, L1)
                return (im, {})

        def _read_one_tag(self):
            """
            Return (True, loc, size, T, L1) if an image that we can read.
            Return (False, loc, size, T, L1) if any other tag.
            """
            head = self._fp_read(6)
            if not head:
                raise IndexError('Reached end of swf movie')
            T, L1, L2 = _swf.get_type_and_len(head)
            if not L2:
                raise RuntimeError('Invalid tag length, could not proceed')
            isimage = False
            sze = L2 - 6
            if T == 0:
                raise IndexError('Reached end of swf movie')
            elif T in [20, 36]:
                isimage = True
            elif T in [6, 21, 35, 90]:
                logger.warning('Ignoring JPEG image: cannot read JPEG.')
            else:
                pass
            return (isimage, sze, T, L1)

        def _get_meta_data(self, index):
            return {}

    class Writer(Format.Writer):

        def _open(self, fps=12, loop=True, html=False, compress=False):
            if not _swf:
                load_lib()
            self._arg_fps = int(fps)
            self._arg_loop = bool(loop)
            self._arg_html = bool(html)
            self._arg_compress = bool(compress)
            self._fp = self.request.get_file()
            self._framecounter = 0
            self._framesize = (100, 100)
            if self._arg_compress:
                self._fp_real = self._fp
                self._fp = BytesIO()

        def _close(self):
            self._complete()
            sze = self._fp.tell()
            self._fp.seek(self._location_to_save_nframes)
            self._fp.write(_swf.int2uint16(self._framecounter))
            if self._arg_compress:
                bb = self._fp.getvalue()
                self._fp = self._fp_real
                self._fp.write(bb[:8])
                self._fp.write(zlib.compress(bb[8:]))
                sze = self._fp.tell()
            self._fp.seek(4)
            self._fp.write(_swf.int2uint32(sze))
            self._fp = None
            if self._arg_html and os.path.isfile(self.request.filename):
                dirname, fname = os.path.split(self.request.filename)
                filename = os.path.join(dirname, fname[:-4] + '.html')
                w, h = self._framesize
                html = HTML % (fname, w, h, fname)
                with open(filename, 'wb') as f:
                    f.write(html.encode('utf-8'))

        def _write_header(self, framesize, fps):
            self._framesize = framesize
            bb = b''
            bb += 'FC'[self._arg_compress].encode('ascii')
            bb += 'WS'.encode('ascii')
            bb += _swf.int2uint8(8)
            bb += '0000'.encode('ascii')
            bb += _swf.Tag().make_rect_record(0, framesize[0], 0, framesize[1]).tobytes()
            bb += _swf.int2uint8(0) + _swf.int2uint8(fps)
            self._location_to_save_nframes = len(bb)
            bb += '00'.encode('ascii')
            self._fp.write(bb)
            taglist = (_swf.FileAttributesTag(), _swf.SetBackgroundTag(0, 0, 0))
            for tag in taglist:
                self._fp.write(tag.get_tag())

        def _complete(self):
            if not self._framecounter:
                self._write_header((10, 10), self._arg_fps)
            if not self._arg_loop:
                self._fp.write(_swf.DoActionTag('stop').get_tag())
            self._fp.write('\x00\x00'.encode('ascii'))

        def _append_data(self, im, meta):
            if im.ndim == 3 and im.shape[-1] == 1:
                im = im[:, :, 0]
            im = image_as_uint(im, bitdepth=8)
            wh = (im.shape[1], im.shape[0])
            isfirstframe = False
            if self._framecounter == 0:
                isfirstframe = True
                self._write_header(wh, self._arg_fps)
            bm = _swf.BitmapTag(im)
            sh = _swf.ShapeTag(bm.id, (0, 0), wh)
            po = _swf.PlaceObjectTag(1, sh.id, move=not isfirstframe)
            sf = _swf.ShowFrameTag()
            for tag in [bm, sh, po, sf]:
                self._fp.write(tag.get_tag())
            self._framecounter += 1

        def set_meta_data(self, meta):
            pass