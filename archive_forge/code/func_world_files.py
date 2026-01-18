import collections
from pathlib import Path
import numpy as np
from PIL import Image
import shapely.geometry as sgeom
@staticmethod
def world_files(fname):
    """
        Determine potential world filename combinations, without checking
        their existence.

        For example, a ``'*.tif'`` file may have one of the following
        popular conventions for world file extensions ``'*.tifw'``,
        ``'*.tfw'``, ``'*.TIFW'`` or ``'*.TFW'``.

        Given the possible world file extensions, the upper case basename
        combinations are also generated. For example, the file 'map.tif'
        will generate the following world file variations, 'map.tifw',
        'map.tfw', 'map.TIFW', 'map.TFW', 'MAP.tifw', 'MAP.tfw', 'MAP.TIFW'
        and 'MAP.TFW'.

        Parameters
        ----------
        fname
            Name of the file for which to get all the possible world
            filename combinations.

        Returns
        -------
            A list of possible world filename combinations.

        Examples
        --------

        >>> from cartopy.io.img_nest import Img
        >>> Img.world_files('img.png')[:6]
        ['img.pngw', 'img.pgw', 'img.PNGW', 'img.PGW', 'IMG.pngw', 'IMG.pgw']
        >>> Img.world_files('/path/to/img.TIF')[:2]
        ['/path/to/img.tifw', '/path/to/img.tfw']
        >>> Img.world_files('/path/to/img/with_no_extension')[0]
        '/path/to/img/with_no_extension.w'

        """
    path = Path(fname)
    fext = path.suffix[1:].lower()
    if len(fext) < 3:
        result = [path.with_suffix('.w'), path.with_suffix('.W')]
    else:
        fext_types = [f'.{fext}w', f'.{fext[0]}{fext[-1]}w']
        fext_types.extend([ext.upper() for ext in fext_types])
        result = [path.with_suffix(ext) for ext in fext_types]
    result += [p.with_name(p.stem.swapcase() + p.suffix) for p in result]
    return [str(r) for r in result]