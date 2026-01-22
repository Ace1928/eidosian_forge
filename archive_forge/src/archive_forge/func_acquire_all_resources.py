import io
import itertools
from pathlib import Path
from urllib.error import HTTPError
import shapefile
import shapely.geometry as sgeom
from cartopy import config
from cartopy.io import Downloader
def acquire_all_resources(self, format_dict):
    from zipfile import ZipFile
    url = self.url(format_dict)
    try:
        shapefile_online = self._urlopen(url)
    except HTTPError:
        try:
            '\n                case if GSHHS has had an update\n                without changing the naming convention\n                '
            url = f'https://www.ngdc.noaa.gov/mgg/shorelines/data/gshhs/oldversions/version{self.gshhs_version}/gshhg-shp-{self.gshhs_version}.zip'
            shapefile_online = self._urlopen(url)
        except HTTPError:
            '\n                case if GSHHS has had an update\n                with changing the naming convention\n                '
            url = 'https://www.ngdc.noaa.gov/mgg/shorelines/data/gshhs/oldversions/version2.3.6/gshhg-shp-2.3.6.zip'
            shapefile_online = self._urlopen(url)
    zfh = ZipFile(io.BytesIO(shapefile_online.read()), 'r')
    shapefile_online.close()
    modified_format_dict = dict(format_dict)
    scales = ('c', 'l', 'i', 'h', 'f')
    levels = (1, 2, 3, 4, 5, 6)
    for scale, level in itertools.product(scales, levels):
        if scale == 'c' and level == 4:
            continue
        modified_format_dict.update({'scale': scale, 'level': level})
        target_path = self.target_path(modified_format_dict)
        target_dir = target_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        for member_path in self.zip_file_contents(modified_format_dict):
            member = zfh.getinfo(member_path.replace('\\', '/'))
            with open(target_path.with_suffix(Path(member_path).suffix), 'wb') as fh:
                fh.write(zfh.open(member).read())
    zfh.close()