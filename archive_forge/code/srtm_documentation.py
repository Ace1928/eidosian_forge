import io
from pathlib import Path
import warnings
import numpy as np
from cartopy import config
import cartopy.crs as ccrs
from cartopy.io import Downloader, LocatedImage, RasterSource, fh_getter

        Return a typical downloader for this class. In general, this static
        method is used to create the default configuration in cartopy.config

        