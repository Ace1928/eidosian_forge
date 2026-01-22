import collections
from pathlib import Path
import numpy as np
from PIL import Image
import shapely.geometry as sgeom
class NestedImageCollection:

    def __init__(self, name, crs, collections, _ancestry=None):
        """
        Represents a complex nest of ImageCollections.

        On construction, the image collections are scanned for ancestry,
        leading to fast image finding capabilities.

        A complex (and time consuming to create) NestedImageCollection instance
        can be saved as a pickle file and subsequently be (quickly) restored.

        There is a simplified creation interface for NestedImageCollection
        ``from_configuration`` for more detail.

        Parameters
        ----------
        name
            The name of the nested image collection.
        crs
            The native :class:`~cartopy.crs.Projection` of all the image
            collections.
        collections
            A list of one or more :class:`~cartopy.io.img_nest.ImageCollection`
            instances.

        """
        _names = {collection.name for collection in collections}
        assert len(_names) == len(collections), 'The collections must have unique names.'
        self.name = name
        self.crs = crs
        self._collections_by_name = {collection.name: collection for collection in collections}

        def sort_func(c):
            return np.max([image.bbox().area for image in c.images])
        self._collections = sorted(collections, key=sort_func, reverse=True)
        self._ancestry = {}
        '\n        maps (collection name, image) to a list of children\n        (collection name, image).\n        '
        if _ancestry is not None:
            self._ancestry = _ancestry
        else:
            parent_wth_children = zip(self._collections, self._collections[1:])
            for parent_collection, collection in parent_wth_children:
                for parent_image in parent_collection.images:
                    for image in collection.images:
                        if self._is_parent(parent_image, image):
                            key = (parent_collection.name, parent_image)
                            self._ancestry.setdefault(key, []).append((collection.name, image))

    @staticmethod
    def _is_parent(parent, child):
        """
        Return whether the given Image is the parent of image.
        Used by __init__.

        """
        result = False
        pbox = parent.bbox()
        cbox = child.bbox()
        if pbox.area > cbox.area:
            result = pbox.intersects(cbox) and (not pbox.touches(cbox))
        return result

    def image_for_domain(self, target_domain, target_z):
        """
        Determine the image that provides complete coverage of target
        location.

        The composed image is merged from one or more image tiles that overlay
        the target location and provide complete image coverage of the target
        location.

        Parameters
        ----------
        target_domain
            A :class:`~shapely.geometry.linestring.LineString`
            instance that specifies the target location requiring image
            coverage.
        target_z
            The name of the target :class`~cartopy.io.img_nest.ImageCollection`
            which specifies the target zoom level (resolution) of the required
            images.

        Returns
        -------
        img, extent, origin
            A tuple containing three items, consisting of the target
            location :class:`numpy.ndarray` image data, the
            (x-lower, x-upper, y-lower, y-upper) extent of the image, and the
            origin for the target location.

        """
        if target_z not in self._collections_by_name:
            raise ValueError(f'{target_z!r} is not one of the possible collections.')
        tiles = []
        for tile in self.find_images(target_domain, target_z):
            try:
                img, extent, origin = self.get_image(tile)
            except OSError:
                continue
            img = np.array(img)
            x = np.linspace(extent[0], extent[1], img.shape[1], endpoint=False)
            y = np.linspace(extent[2], extent[3], img.shape[0], endpoint=False)
            tiles.append([np.array(img), x, y, origin])
        from cartopy.io.img_tiles import _merge_tiles
        img, extent, origin = _merge_tiles(tiles)
        return (img, extent, origin)

    def find_images(self, target_domain, target_z, start_tiles=None):
        """
        A generator that finds all images that overlap the bounded
        target location.

        Parameters
        ----------
        target_domain
            A :class:`~shapely.geometry.linestring.LineString` instance that
            specifies the target location requiring image coverage.

        target_z
            The name of the target
            :class:`~cartopy.io.img_nest.ImageCollection` which specifies
            the target zoom level (resolution) of the required images.
        start_tiles: optional
            A list of one or more tuple pairs, composed of a
            :class:`~cartopy.io.img_nest.ImageCollection` name and an
            :class:`~cartopy.io.img_nest.Img` instance, from which to search
            for the target images.

        Returns
        -------
        generator
            A generator tuple pair composed of a
            :class:`~cartopy.io.img_nest.ImageCollection` name and an
            :class:`~cartopy.io.img_nest.Img` instance.

        """
        if target_z not in self._collections_by_name:
            raise ValueError(f'{target_z!r} is not one of the possible collections.')
        if start_tiles is None:
            start_tiles = ((self._collections[0].name, img) for img in self._collections[0].images)
        for start_tile in start_tiles:
            domain = start_tile[1].bbox()
            if target_domain.intersects(domain) and (not target_domain.touches(domain)):
                if start_tile[0] == target_z:
                    yield start_tile
                else:
                    for tile in self.subtiles(start_tile):
                        yield from self.find_images(target_domain, target_z, start_tiles=[tile])

    def subtiles(self, collection_image):
        """
        Find the higher resolution image tiles that compose this parent
        image tile.

        Parameters
        ----------
        collection_image
            A tuple pair containing the parent
            :class:`~cartopy.io.img_nest.ImageCollection` name and
            :class:`~cartopy.io.img_nest.Img` instance.

        Returns
        -------
        iterator
            An iterator of tuple pairs containing the higher resolution child
            :class:`~cartopy.io.img_nest.ImageCollection` name and
            :class:`~cartopy.io.img_nest.Img` instance that compose the parent.

        """
        return iter(self._ancestry.get(collection_image, []))
    desired_tile_form = 'RGB'

    def get_image(self, collection_image):
        """
        Retrieve the data of the target image from file.

        Parameters
        ----------
        collection_image:
            A tuple pair containing the target
            :class:`~cartopy.io.img_nest.ImageCollection` name and
            :class:`~cartopy.io.img_nest.Img` instance.

        Returns
        -------
        img_data, img.extent, img.origin
            A tuple containing three items, consisting of the associated image
            file data, the (x_lower, x_upper, y_lower, y_upper) extent of the
            image, and the image origin.

        Note
        ----
          The format of the retrieved image file data is controlled by
          :attr:`~cartopy.io.img_nest.NestedImageCollection.desired_tile_form`,
          which defaults to 'RGB' format.

        """
        img = collection_image[1]
        img_data = Image.open(img.filename)
        img_data = img_data.convert(self.desired_tile_form)
        return (img_data, img.extent, img.origin)

    @classmethod
    def from_configuration(cls, name, crs, name_dir_pairs, glob_pattern='*.tif', img_class=Img):
        """
        Create a :class:`~cartopy.io.img_nest.NestedImageCollection` instance
        given the list of image collection name and directory path pairs.

        This is very convenient functionality for simple configuration level
        creation of this complex object.

        For example, to produce a nested collection of OS map tiles::

            files = [['OS 1:1,000,000', '/directory/to/1_to_1m'],
                     ['OS 1:250,000', '/directory/to/1_to_250k'],
                     ['OS 1:50,000', '/directory/to/1_to_50k'],
                    ]
            r = NestedImageCollection.from_configuration('os',
                                                         ccrs.OSGB(),
                                                         files)

        Parameters
        ----------
        name
            The name for the
            :class:`~cartopy.io.img_nest.NestedImageCollection` instance.
        crs
            The :class:`~cartopy.crs.Projection` of the image collection.
        name_dir_pairs
            A list of image collection name and directory path pairs.
        glob_pattern: optional
            The image collection filename glob pattern. Defaults
            to ``'*.tif'``.
        img_class: optional
            The class of images created in the image collection.

        Returns
        -------
        A :class:`~cartopy.io.img_nest.NestedImageCollection` instance.

        Warnings
        --------
            The list of image collection name and directory path pairs must be
            given in increasing resolution order i.e. from low resolution to
            high resolution.

        """
        collections = []
        for collection_name, collection_dir in name_dir_pairs:
            collection = ImageCollection(collection_name, crs)
            collection.scan_dir_for_imgs(collection_dir, glob_pattern=glob_pattern, img_class=img_class)
            collections.append(collection)
        return cls(name, crs, collections)