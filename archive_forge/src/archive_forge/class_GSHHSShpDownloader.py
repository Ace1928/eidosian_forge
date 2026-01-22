import io
import itertools
from pathlib import Path
from urllib.error import HTTPError
import shapefile
import shapely.geometry as sgeom
from cartopy import config
from cartopy.io import Downloader
class GSHHSShpDownloader(Downloader):
    """
    Specialise :class:`cartopy.io.Downloader` to download the zipped
    GSHHS shapefiles and extract them to the defined location.

    The keys which should be passed through when using the ``format_dict``
    are ``scale`` (a single character indicating the resolution) and ``level``
    (a number indicating the type of feature).

    """
    FORMAT_KEYS = ('config', 'scale', 'level')
    gshhs_version = '2.3.7'
    _GSHHS_URL_TEMPLATE = f'https://www.ngdc.noaa.gov/mgg/shorelines/data/gshhs/latest/gshhg-shp-{gshhs_version}.zip'

    def __init__(self, url_template=_GSHHS_URL_TEMPLATE, target_path_template=None, pre_downloaded_path_template=''):
        super().__init__(url_template, target_path_template, pre_downloaded_path_template)

    def zip_file_contents(self, format_dict):
        """
        Return a generator of the filenames to be found in the downloaded
        GSHHS zip file for the specified resource.

        """
        for ext in ['.shp', '.dbf', '.shx']:
            p = Path('GSHHS_shp', '{scale}', 'GSHHS_{scale}_L{level}{extension}')
            yield str(p).format(extension=ext, **format_dict)

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

    def acquire_resource(self, target_path, format_dict):
        """
        Download the zip file and extracts the files listed in
        :meth:`zip_file_contents` to the target path.

        Note
        ----
            Because some of the GSHSS data is available with the cartopy
            repository, scales of "l" or "c" will not be downloaded if they
            exist in the ``cartopy.config['repo_data_dir']`` directory.

        """
        repo_fname_pattern = str(Path('shapefiles') / 'gshhs' / '{scale}' / 'GSHHS_{scale}_L?.shp')
        repo_fname_pattern = repo_fname_pattern.format(**format_dict)
        repo_fnames = list(config['repo_data_dir'].glob(repo_fname_pattern))
        if repo_fnames:
            assert len(repo_fnames) == 1, '>1 repo files found for GSHHS'
            return repo_fnames[0]
        self.acquire_all_resources(format_dict)
        if not target_path.exists():
            raise RuntimeError(f'Failed to download and extract GSHHS shapefile to {target_path!r}.')
        return target_path

    @staticmethod
    def default_downloader():
        """
        Return a GSHHSShpDownloader instance that expects (and if necessary
        downloads and installs) shapefiles in the data directory of the
        cartopy installation.

        Typically, a user will not need to call this staticmethod.

        To find the path template of the GSHHSShpDownloader:

            >>> gshhs_dnldr = GSHHSShpDownloader.default_downloader()
            >>> print(gshhs_dnldr.target_path_template)
            {config[data_dir]}/shapefiles/gshhs/{scale}/GSHHS_{scale}_L{level}.shp

        """
        default_spec = ('shapefiles', 'gshhs', '{scale}', 'GSHHS_{scale}_L{level}.shp')
        gshhs_path_template = str(Path('{config[data_dir]}').joinpath(*default_spec))
        pre_path_tmplt = str(Path('{config[pre_existing_data_dir]}').joinpath(*default_spec))
        return GSHHSShpDownloader(target_path_template=gshhs_path_template, pre_downloaded_path_template=pre_path_tmplt)