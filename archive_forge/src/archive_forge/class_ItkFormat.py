from ..core import Format, has_module
class ItkFormat(Format):
    """See :mod:`imageio.plugins.simpleitk`"""

    def _can_read(self, request):
        if request.extension in ITK_FORMATS:
            return True
        if has_module('itk.ImageIOBase') or has_module('SimpleITK'):
            return request.extension in ALL_FORMATS

    def _can_write(self, request):
        if request.extension in ITK_FORMATS:
            return True
        if has_module('itk.ImageIOBase') or has_module('SimpleITK'):
            return request.extension in ALL_FORMATS

    class Reader(Format.Reader):

        def _open(self, pixel_type=None, fallback_only=None, **kwargs):
            if not _itk:
                load_lib()
            args = ()
            if pixel_type is not None:
                args += (pixel_type,)
                if fallback_only is not None:
                    args += (fallback_only,)
            self._img = _read_function(self.request.get_local_filename(), *args)

        def _get_length(self):
            return 1

        def _close(self):
            pass

        def _get_data(self, index):
            if index != 0:
                error_msg = 'Index out of range while reading from itk file'
                raise IndexError(error_msg)
            return (_itk.GetArrayFromImage(self._img), {})

        def _get_meta_data(self, index):
            error_msg = 'The itk plugin does not support meta data, currently.'
            raise RuntimeError(error_msg)

    class Writer(Format.Writer):

        def _open(self):
            if not _itk:
                load_lib()

        def _close(self):
            pass

        def _append_data(self, im, meta):
            _itk_img = _itk.GetImageFromArray(im)
            _write_function(_itk_img, self.request.get_local_filename())

        def set_meta_data(self, meta):
            error_msg = 'The itk plugin does not support meta data, currently.'
            raise RuntimeError(error_msg)