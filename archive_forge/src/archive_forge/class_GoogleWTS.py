from abc import ABCMeta, abstractmethod
import concurrent.futures
import io
from pathlib import Path
import warnings
import numpy as np
from PIL import Image
import shapely.geometry as sgeom
import cartopy
import cartopy.crs as ccrs
class GoogleWTS(metaclass=ABCMeta):
    """
    Implement web tile retrieval using the Google WTS coordinate system.

    A "tile" in this class refers to the coordinates (x, y, z).

    The tiles can be saved to a cache directory using the cache parameter, so
    they are downloaded only once. If it is set to True, the default path
    stored in the cartopy.config dictionary is used. If it is set to a custom
    path, this path is used instead of the default one. If it is set to False
    (the default behavior), the tiles are downloaded each time.

    """
    _MAX_THREADS = 24

    def __init__(self, desired_tile_form='RGB', user_agent=f'CartoPy/{cartopy.__version__}', cache=False):
        self.imgs = []
        self.crs = ccrs.Mercator.GOOGLE
        self.desired_tile_form = desired_tile_form
        self.user_agent = user_agent
        self._default_cache = False
        if cache is True:
            self._default_cache = True
            self.cache_path = Path(cartopy.config['cache_dir'])
        elif cache is False:
            self.cache_path = None
        else:
            self.cache_path = Path(cache)
        self.cache = set({})
        self._load_cache()

    def image_for_domain(self, target_domain, target_z):
        tiles = []

        def fetch_tile(tile):
            try:
                img, extent, origin = self.get_image(tile)
            except OSError:
                raise
            img = np.array(img)
            x = np.linspace(extent[0], extent[1], img.shape[1])
            y = np.linspace(extent[2], extent[3], img.shape[0])
            return (img, x, y, origin)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._MAX_THREADS) as executor:
            futures = []
            for tile in self.find_images(target_domain, target_z):
                futures.append(executor.submit(fetch_tile, tile))
            for future in concurrent.futures.as_completed(futures):
                try:
                    img, x, y, origin = future.result()
                    tiles.append([img, x, y, origin])
                except OSError:
                    pass
        img, extent, origin = _merge_tiles(tiles)
        return (img, extent, origin)

    @property
    def _cache_dir(self):
        """Return the name of the cache directory"""
        return self.cache_path / self.__class__.__name__

    def _load_cache(self):
        """Load the cache"""
        if self.cache_path is not None:
            cache_dir = self._cache_dir
            if not cache_dir.exists():
                cache_dir.mkdir(parents=True)
                if self._default_cache:
                    warnings.warn(f'Cartopy created the following directory to cache GoogleWTS tiles: {cache_dir}')
            self.cache = self.cache.union(set(cache_dir.iterdir()))

    def _find_images(self, target_domain, target_z, start_tile=(0, 0, 0)):
        """Target domain is a shapely polygon in native coordinates."""
        assert isinstance(target_z, int) and target_z >= 0, 'target_z must be an integer >=0.'
        x0, x1, y0, y1 = self._tileextent(start_tile)
        domain = sgeom.box(x0, y0, x1, y1)
        if domain.intersects(target_domain):
            if start_tile[2] == target_z:
                yield start_tile
            else:
                for tile in self._subtiles(start_tile):
                    yield from self._find_images(target_domain, target_z, start_tile=tile)
    find_images = _find_images

    def subtiles(self, x_y_z):
        x, y, z = x_y_z
        for xi in range(0, 2):
            for yi in range(0, 2):
                yield (x * 2 + xi, y * 2 + yi, z + 1)
    _subtiles = subtiles

    def tile_bbox(self, x, y, z, y0_at_north_pole=True):
        """
        Return the ``(x0, x1), (y0, y1)`` bounding box for the given x, y, z
        tile position.

        Parameters
        ----------
        x
            The x tile coordinate in the Google tile numbering system.
        y
            The y tile coordinate in the Google tile numbering system.
        z
            The z tile coordinate in the Google tile numbering system.

        y0_at_north_pole: optional
            Boolean representing whether the numbering of the y coordinate
            starts at the north pole (as is the convention for Google tiles)
            or not (in which case it will start at the south pole, as is the
            convention for TMS). Defaults to True.


        """
        n = 2 ** z
        assert 0 <= x <= n - 1, f"Tile's x index is out of range. Upper limit {n}. Got {x}"
        assert 0 <= y <= n - 1, f"Tile's y index is out of range. Upper limit {n}. Got {y}"
        x0, x1 = self.crs.x_limits
        y0, y1 = self.crs.y_limits
        box_h = (y1 - y0) / n
        box_w = (x1 - x0) / n
        n_xs = x0 + (x + np.arange(0, 2, dtype=np.float64)) * box_w
        n_ys = y0 + (y + np.arange(0, 2, dtype=np.float64)) * box_h
        if y0_at_north_pole:
            n_ys = -1 * n_ys[::-1]
        return (n_xs, n_ys)

    def tileextent(self, x_y_z):
        """Return extent tuple ``(x0,x1,y0,y1)`` in Mercator coordinates."""
        x, y, z = x_y_z
        x_lim, y_lim = self.tile_bbox(x, y, z, y0_at_north_pole=True)
        return tuple(x_lim) + tuple(y_lim)
    _tileextent = tileextent

    @abstractmethod
    def _image_url(self, tile):
        pass

    def get_image(self, tile):
        from urllib.request import HTTPError, Request, URLError, urlopen
        if self.cache_path is not None:
            filename = '_'.join([str(i) for i in tile]) + '.npy'
            cached_file = self._cache_dir / filename
        else:
            cached_file = None
        if cached_file in self.cache:
            img = np.load(cached_file, allow_pickle=False)
        else:
            url = self._image_url(tile)
            try:
                request = Request(url, headers={'User-Agent': self.user_agent})
                fh = urlopen(request)
                im_data = io.BytesIO(fh.read())
                fh.close()
                img = Image.open(im_data)
            except (HTTPError, URLError) as err:
                print(err)
                img = Image.fromarray(np.full((256, 256, 3), (250, 250, 250), dtype=np.uint8))
            img = img.convert(self.desired_tile_form)
            if self.cache_path is not None:
                np.save(cached_file, img, allow_pickle=False)
                self.cache.add(cached_file)
        return (img, self.tileextent(tile), 'lower')