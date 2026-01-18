from __future__ import print_function

    Common GLES Subset Extraction Script
    ====================================

    In Kivy, our goal is to use OpenGL ES 2.0 (GLES2) for all drawing on all
    platforms. The problem is that GLES2 is not a proper subset of any OpenGL
    Desktop (GL) version prior to version 4.1.
    However, to keep all our drawing cross-platform compatible, we're
    restricting the Kivy drawing core to a real subset of GLES2 that is
    available on all platforms.

    This script therefore parses the GL and GL Extension (GLEXT) headers and
    compares them with the GLES2 header. It then generates a header that only
    contains symbols that are common to GLES2 and at least either GL or GLEXT.
    However, since GLES2 doesn't support double values, we also need to do some
    renaming, because functions in GL that took doubles as arguments now take
    floats in GLES2, with their function name being suffixed with 'f'.

    Furthermore, sometimes the pure symbol name doesn't match because there
    might be an _EXT or _ARB or something akin to that at the end of a symbol
    name. In that case, we take the symbol from the original header and add
    a #define directive to redirect to that symbol from the symbol name without
    extension.
