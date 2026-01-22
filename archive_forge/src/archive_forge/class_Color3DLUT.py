from __future__ import annotations
import functools
class Color3DLUT(MultibandFilter):
    """Three-dimensional color lookup table.

    Transforms 3-channel pixels using the values of the channels as coordinates
    in the 3D lookup table and interpolating the nearest elements.

    This method allows you to apply almost any color transformation
    in constant time by using pre-calculated decimated tables.

    .. versionadded:: 5.2.0

    :param size: Size of the table. One int or tuple of (int, int, int).
                 Minimal size in any dimension is 2, maximum is 65.
    :param table: Flat lookup table. A list of ``channels * size**3``
                  float elements or a list of ``size**3`` channels-sized
                  tuples with floats. Channels are changed first,
                  then first dimension, then second, then third.
                  Value 0.0 corresponds lowest value of output, 1.0 highest.
    :param channels: Number of channels in the table. Could be 3 or 4.
                     Default is 3.
    :param target_mode: A mode for the result image. Should have not less
                        than ``channels`` channels. Default is ``None``,
                        which means that mode wouldn't be changed.
    """
    name = 'Color 3D LUT'

    def __init__(self, size, table, channels=3, target_mode=None, **kwargs):
        if channels not in (3, 4):
            msg = 'Only 3 or 4 output channels are supported'
            raise ValueError(msg)
        self.size = size = self._check_size(size)
        self.channels = channels
        self.mode = target_mode
        copy_table = kwargs.get('_copy_table', True)
        items = size[0] * size[1] * size[2]
        wrong_size = False
        numpy = None
        if hasattr(table, 'shape'):
            try:
                import numpy
            except ImportError:
                pass
        if numpy and isinstance(table, numpy.ndarray):
            if copy_table:
                table = table.copy()
            if table.shape in [(items * channels,), (items, channels), (size[2], size[1], size[0], channels)]:
                table = table.reshape(items * channels)
            else:
                wrong_size = True
        else:
            if copy_table:
                table = list(table)
            if table and isinstance(table[0], (list, tuple)):
                table, raw_table = ([], table)
                for pixel in raw_table:
                    if len(pixel) != channels:
                        msg = f'The elements of the table should have a length of {channels}.'
                        raise ValueError(msg)
                    table.extend(pixel)
        if wrong_size or len(table) != items * channels:
            msg = f'The table should have either channels * size**3 float items or size**3 items of channels-sized tuples with floats. Table should be: {channels}x{size[0]}x{size[1]}x{size[2]}. Actual length: {len(table)}'
            raise ValueError(msg)
        self.table = table

    @staticmethod
    def _check_size(size):
        try:
            _, _, _ = size
        except ValueError as e:
            msg = 'Size should be either an integer or a tuple of three integers.'
            raise ValueError(msg) from e
        except TypeError:
            size = (size, size, size)
        size = [int(x) for x in size]
        for size_1d in size:
            if not 2 <= size_1d <= 65:
                msg = 'Size should be in [2, 65] range.'
                raise ValueError(msg)
        return size

    @classmethod
    def generate(cls, size, callback, channels=3, target_mode=None):
        """Generates new LUT using provided callback.

        :param size: Size of the table. Passed to the constructor.
        :param callback: Function with three parameters which correspond
                         three color channels. Will be called ``size**3``
                         times with values from 0.0 to 1.0 and should return
                         a tuple with ``channels`` elements.
        :param channels: The number of channels which should return callback.
        :param target_mode: Passed to the constructor of the resulting
                            lookup table.
        """
        size_1d, size_2d, size_3d = cls._check_size(size)
        if channels not in (3, 4):
            msg = 'Only 3 or 4 output channels are supported'
            raise ValueError(msg)
        table = [0] * (size_1d * size_2d * size_3d * channels)
        idx_out = 0
        for b in range(size_3d):
            for g in range(size_2d):
                for r in range(size_1d):
                    table[idx_out:idx_out + channels] = callback(r / (size_1d - 1), g / (size_2d - 1), b / (size_3d - 1))
                    idx_out += channels
        return cls((size_1d, size_2d, size_3d), table, channels=channels, target_mode=target_mode, _copy_table=False)

    def transform(self, callback, with_normals=False, channels=None, target_mode=None):
        """Transforms the table values using provided callback and returns
        a new LUT with altered values.

        :param callback: A function which takes old lookup table values
                         and returns a new set of values. The number
                         of arguments which function should take is
                         ``self.channels`` or ``3 + self.channels``
                         if ``with_normals`` flag is set.
                         Should return a tuple of ``self.channels`` or
                         ``channels`` elements if it is set.
        :param with_normals: If true, ``callback`` will be called with
                             coordinates in the color cube as the first
                             three arguments. Otherwise, ``callback``
                             will be called only with actual color values.
        :param channels: The number of channels in the resulting lookup table.
        :param target_mode: Passed to the constructor of the resulting
                            lookup table.
        """
        if channels not in (None, 3, 4):
            msg = 'Only 3 or 4 output channels are supported'
            raise ValueError(msg)
        ch_in = self.channels
        ch_out = channels or ch_in
        size_1d, size_2d, size_3d = self.size
        table = [0] * (size_1d * size_2d * size_3d * ch_out)
        idx_in = 0
        idx_out = 0
        for b in range(size_3d):
            for g in range(size_2d):
                for r in range(size_1d):
                    values = self.table[idx_in:idx_in + ch_in]
                    if with_normals:
                        values = callback(r / (size_1d - 1), g / (size_2d - 1), b / (size_3d - 1), *values)
                    else:
                        values = callback(*values)
                    table[idx_out:idx_out + ch_out] = values
                    idx_in += ch_in
                    idx_out += ch_out
        return type(self)(self.size, table, channels=ch_out, target_mode=target_mode or self.mode, _copy_table=False)

    def __repr__(self):
        r = [f'{self.__class__.__name__} from {self.table.__class__.__name__}', 'size={:d}x{:d}x{:d}'.format(*self.size), f'channels={self.channels:d}']
        if self.mode:
            r.append(f'target_mode={self.mode}')
        return '<{}>'.format(' '.join(r))

    def filter(self, image):
        from . import Image
        return image.color_lut_3d(self.mode or image.mode, Image.Resampling.BILINEAR, self.channels, self.size[0], self.size[1], self.size[2], self.table)