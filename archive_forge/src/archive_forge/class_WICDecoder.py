from pyglet.image import *
from pyglet.image.codecs import *
from pyglet.libs.win32 import _kernel32 as kernel32
from pyglet.libs.win32 import _ole32 as ole32
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
class WICDecoder(ImageDecoder):
    """Windows Imaging Component.
    This decoder is a replacement for GDI and GDI+ starting with Windows 7 with more features up to Windows 10."""

    def __init__(self):
        super(ImageDecoder, self).__init__()
        self._factory = _factory

    def get_file_extensions(self):
        return ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.ico', '.jxr', '.hdp', '.wdp']

    def _load_bitmap_decoder(self, filename, file):
        data = file.read()
        hglob = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(data))
        ptr = kernel32.GlobalLock(hglob)
        memmove(ptr, data, len(data))
        kernel32.GlobalUnlock(hglob)
        stream = com.pIUnknown()
        ole32.CreateStreamOnHGlobal(hglob, True, byref(stream))
        decoder = IWICBitmapDecoder()
        status = self._factory.CreateDecoderFromStream(stream, None, WICDecodeMetadataCacheOnDemand, byref(decoder))
        if status != 0:
            stream.Release()
            raise ImageDecodeException('WIC cannot load %r' % (filename or file))
        return (decoder, stream)

    def _get_bitmap_frame(self, bitmap_decoder, frame_index):
        bitmap = IWICBitmapFrameDecode()
        bitmap_decoder.GetFrame(frame_index, byref(bitmap))
        return bitmap

    def get_image(self, bitmap, target_fmt=GUID_WICPixelFormat32bppBGRA):
        """Get's image from bitmap, specifying target format, bitmap is released before returning."""
        width = UINT()
        height = UINT()
        bitmap.GetSize(byref(width), byref(height))
        width = int(width.value)
        height = int(height.value)
        pf = com.GUID(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        bitmap.GetPixelFormat(byref(pf))
        fmt = 'BGRA'
        if pf != target_fmt:
            converter = IWICFormatConverter()
            self._factory.CreateFormatConverter(byref(converter))
            conversion_possible = BOOL()
            converter.CanConvert(pf, target_fmt, byref(conversion_possible))
            if not conversion_possible:
                target_fmt = GUID_WICPixelFormat24bppBGR
                fmt = 'BGR'
            converter.Initialize(bitmap, target_fmt, WICBitmapDitherTypeNone, None, 0, WICBitmapPaletteTypeCustom)
            bitmap.Release()
            bitmap = converter
        flipper = IWICBitmapFlipRotator()
        self._factory.CreateBitmapFlipRotator(byref(flipper))
        flipper.Initialize(bitmap, WICBitmapTransformFlipVertical)
        stride = len(fmt) * width
        buffer_size = stride * height
        buffer = (BYTE * buffer_size)()
        flipper.CopyPixels(None, stride, buffer_size, byref(buffer))
        flipper.Release()
        bitmap.Release()
        return ImageData(width, height, fmt, buffer)

    def _delete_bitmap_decoder(self, bitmap_decoder, stream):
        bitmap_decoder.Release()
        stream.Release()

    def decode(self, filename, file):
        if not file:
            file = open(filename, 'rb')
        bitmap_decoder, stream = self._load_bitmap_decoder(filename, file)
        bitmap = self._get_bitmap_frame(bitmap_decoder, 0)
        image = self.get_image(bitmap)
        self._delete_bitmap_decoder(bitmap_decoder, stream)
        return image

    @staticmethod
    def get_property_value(reader, metadata_name):
        """
            Uses a metadata name and reader to return a single value. Can be used to get metadata from images.
            If failure, returns 0.
            Also handles cleanup of PROPVARIANT.
        """
        try:
            prop = PROPVARIANT()
            reader.GetMetadataByName(metadata_name, byref(prop))
            value = prop.llVal
            ole32.PropVariantClear(byref(prop))
        except OSError:
            value = 0
        return value