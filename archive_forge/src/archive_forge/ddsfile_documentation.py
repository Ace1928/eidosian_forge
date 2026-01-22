from struct import pack, unpack, calcsize

DDS File library
================

This library can be used to parse and save DDS
(`DirectDraw Surface <https://en.wikipedia.org/wiki/DirectDraw_Surface>`)
files.

The initial version was written by::

    Alexey Borzenkov (snaury@gmail.com)

All the initial work credits go to him! Thank you :)

This version uses structs instead of ctypes.


DDS Format
----------

::

    [DDS ][SurfaceDesc][Data]

    [SurfaceDesc]:: (everything is uint32)
        Size
        Flags
        Height
        Width
        PitchOrLinearSize
        Depth
        MipmapCount
        Reserved1 * 11
        [PixelFormat]::
            Size
            Flags
            FourCC
            RGBBitCount
            RBitMask
            GBitMask
            BBitMask
            ABitMask
        [Caps]::
            Caps1
            Caps2
            Reserved1 * 2
        Reserverd2

.. warning::

    This is an external library and Kivy does not provide any support for it.
    It might change in the future and we advise you don't rely on it in your
    code.

