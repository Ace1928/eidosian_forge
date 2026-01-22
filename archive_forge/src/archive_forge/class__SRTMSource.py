import io
from pathlib import Path
import warnings
import numpy as np
from cartopy import config
import cartopy.crs as ccrs
from cartopy.io import Downloader, LocatedImage, RasterSource, fh_getter
class _SRTMSource(RasterSource):
    """
    A source of SRTM data, which implements Cartopy's :ref:`RasterSource
    interface <raster-source-interface>`.

    """

    def __init__(self, resolution, downloader, max_nx, max_ny):
        """
        Parameters
        ----------
        resolution
            The resolution of SRTM to download, in integer arc-seconds.
            Data is available at resolutions of 3 and 1 arc-seconds.
        downloader: :class:`cartopy.io.Downloader` instance or None
            The downloader to use for the SRTM dataset. If None, the
            downloader will be taken using
            :class:`cartopy.io.Downloader.from_config` with ('SRTM',
            'SRTM{resolution}') as the target.
        max_nx
            The maximum number of x tiles to be combined when producing a
            wider composite for this RasterSource.
        max_ny
            The maximum number of y tiles to be combined when producing a
            taller composite for this RasterSource.

        """
        if resolution == 3:
            self._shape = (1201, 1201)
        elif resolution == 1:
            self._shape = (3601, 3601)
        else:
            raise ValueError(f'Resolution is an unexpected value ({resolution}).')
        self._resolution = resolution
        self.crs = ccrs.PlateCarree()
        self.downloader = downloader
        if self.downloader is None:
            self.downloader = Downloader.from_config(('SRTM', f'SRTM{resolution}'))
        self._max_tiles = (max_nx, max_ny)

    def validate_projection(self, projection):
        return projection == self.crs

    def fetch_raster(self, projection, extent, target_resolution):
        """
        Fetch SRTM elevation for the given projection and approximate extent.

        """
        if not self.validate_projection(projection):
            raise ValueError(f'Unsupported projection for the SRTM{self._resolution} source.')
        min_x, max_x, min_y, max_y = extent
        min_x, min_y = np.floor([min_x, min_y])
        nx = int(np.ceil(max_x) - min_x)
        ny = int(np.ceil(max_y) - min_y)
        skip = False
        if nx > self._max_tiles[0]:
            warnings.warn(f'Required SRTM{self._resolution} tile count ({nx}) exceeds maximum ({self._max_tiles[0]}). Increase max_nx limit.')
            skip = True
        if ny > self._max_tiles[1]:
            warnings.warn(f'Required SRTM{self._resolution} tile count ({ny}) exceeds maximum ({self._max_tiles[1]}). Increase max_ny limit.')
            skip = True
        if skip:
            return []
        else:
            img, _, extent = self.combined(min_x, min_y, nx, ny)
            return [LocatedImage(np.flipud(img), extent)]

    def srtm_fname(self, lon, lat):
        """
        Return the filename for the given lon/lat SRTM tile (downloading if
        necessary), or None if no such tile exists (i.e. the tile would be
        entirely over water, or out of latitude range).

        """
        if int(lon) != lon or int(lat) != lat:
            raise ValueError('Integer longitude/latitude values required.')
        x = '%s%03d' % ('E' if lon >= 0 else 'W', abs(int(lon)))
        y = '%s%02d' % ('N' if lat >= 0 else 'S', abs(int(lat)))
        srtm_downloader = Downloader.from_config(('SRTM', f'SRTM{self._resolution}'))
        params = {'config': config, 'resolution': self._resolution, 'x': x, 'y': y}
        if srtm_downloader.url(params) is None:
            return None
        else:
            return self.downloader.path(params)

    def combined(self, lon_min, lat_min, nx, ny):
        """
        Return an image and its extent for the group of nx by ny tiles
        starting at the given bottom left location.

        """
        bottom_left_ll = (lon_min, lat_min)
        shape = np.array(self._shape)
        img = np.zeros(shape * (ny, nx))
        for i, j in np.ndindex(nx, ny):
            x_img_slice = slice(i * shape[1], (i + 1) * shape[1])
            y_img_slice = slice(j * shape[0], (j + 1) * shape[0])
            try:
                tile_img, _, _ = self.single_tile(bottom_left_ll[0] + i, bottom_left_ll[1] + j)
            except ValueError:
                img[y_img_slice, x_img_slice] = 0
            else:
                img[y_img_slice, x_img_slice] = tile_img
        extent = (bottom_left_ll[0], bottom_left_ll[0] + nx, bottom_left_ll[1], bottom_left_ll[1] + ny)
        return (img, self.crs, extent)

    def single_tile(self, lon, lat):
        fname = self.srtm_fname(lon, lat)
        if fname is None:
            raise ValueError('No srtm tile found for those coordinates.')
        return read_SRTM(fname)