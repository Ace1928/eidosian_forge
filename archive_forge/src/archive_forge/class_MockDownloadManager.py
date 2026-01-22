import os
import re
import urllib.parse
from pathlib import Path
from typing import Callable, List, Optional, Union
from zipfile import ZipFile
from ..utils.file_utils import cached_path, hf_github_url
from ..utils.logging import get_logger
from ..utils.version import Version
class MockDownloadManager:
    dummy_file_name = 'dummy_data'
    datasets_scripts_dir = 'datasets'
    is_streaming = False

    def __init__(self, dataset_name: str, config: str, version: Union[Version, str], cache_dir: Optional[str]=None, use_local_dummy_data: bool=False, load_existing_dummy_data: bool=True, download_callbacks: Optional[List[Callable]]=None):
        self.downloaded_size = 0
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.use_local_dummy_data = use_local_dummy_data
        self.config = config
        self.download_callbacks: List[Callable] = download_callbacks or []
        self.load_existing_dummy_data = load_existing_dummy_data
        self.version_name = str(version)
        self._dummy_file = None
        self._bucket_url = None

    @property
    def dummy_file(self):
        if self._dummy_file is None:
            self._dummy_file = self.download_dummy_data()
        return self._dummy_file

    @property
    def dummy_data_folder(self):
        if self.config is not None:
            return os.path.join('dummy', self.config.name, self.version_name)
        return os.path.join('dummy', self.version_name)

    @property
    def dummy_zip_file(self):
        return os.path.join(self.dummy_data_folder, 'dummy_data.zip')

    def download_dummy_data(self):
        path_to_dummy_data_dir = self.local_path_to_dummy_data if self.use_local_dummy_data is True else self.github_path_to_dummy_data
        local_path = cached_path(path_to_dummy_data_dir, cache_dir=self.cache_dir, extract_compressed_file=True, force_extract=True)
        return os.path.join(local_path, self.dummy_file_name)

    @property
    def local_path_to_dummy_data(self):
        return os.path.join(self.datasets_scripts_dir, self.dataset_name, self.dummy_zip_file)

    @property
    def github_path_to_dummy_data(self):
        if self._bucket_url is None:
            self._bucket_url = hf_github_url(self.dataset_name, self.dummy_zip_file.replace(os.sep, '/'))
        return self._bucket_url

    @property
    def manual_dir(self):
        if os.path.isdir(self.dummy_file):
            return self.dummy_file
        return '/'.join(self.dummy_file.replace(os.sep, '/').split('/')[:-1])

    def download_and_extract(self, data_url, *args):
        if self.load_existing_dummy_data:
            dummy_file = self.dummy_file
        else:
            dummy_file = self.dummy_file_name
        if isinstance(data_url, dict):
            return self.create_dummy_data_dict(dummy_file, data_url)
        elif isinstance(data_url, (list, tuple)):
            return self.create_dummy_data_list(dummy_file, data_url)
        else:
            return self.create_dummy_data_single(dummy_file, data_url)

    def download(self, data_url, *args):
        return self.download_and_extract(data_url)

    def download_custom(self, data_url, custom_download):
        return self.download_and_extract(data_url)

    def extract(self, path, *args, **kwargs):
        return path

    def get_recorded_sizes_checksums(self):
        return {}

    def create_dummy_data_dict(self, path_to_dummy_data, data_url):
        dummy_data_dict = {}
        for key, single_urls in data_url.items():
            for download_callback in self.download_callbacks:
                if isinstance(single_urls, list):
                    for single_url in single_urls:
                        download_callback(single_url)
                else:
                    single_url = single_urls
                    download_callback(single_url)
            if isinstance(single_urls, list):
                value = [os.path.join(path_to_dummy_data, urllib.parse.quote_plus(Path(x).name)) for x in single_urls]
            else:
                single_url = single_urls
                value = os.path.join(path_to_dummy_data, urllib.parse.quote_plus(Path(single_url).name))
            dummy_data_dict[key] = value
        if all((isinstance(i, str) for i in dummy_data_dict.values())) and len(set(dummy_data_dict.values())) < len(dummy_data_dict.values()):
            dummy_data_dict = {key: value + key for key, value in dummy_data_dict.items()}
        return dummy_data_dict

    def create_dummy_data_list(self, path_to_dummy_data, data_url):
        dummy_data_list = []
        is_tf_records = all((bool(re.findall('[0-9]{3,}-of-[0-9]{3,}', url)) for url in data_url))
        is_pubmed_records = all((url.startswith('https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed') for url in data_url))
        if data_url and (is_tf_records or is_pubmed_records):
            data_url = [data_url[0]] * len(data_url)
        for single_url in data_url:
            for download_callback in self.download_callbacks:
                download_callback(single_url)
            value = os.path.join(path_to_dummy_data, urllib.parse.quote_plus(single_url.split('/')[-1]))
            dummy_data_list.append(value)
        return dummy_data_list

    def create_dummy_data_single(self, path_to_dummy_data, data_url):
        for download_callback in self.download_callbacks:
            download_callback(data_url)
        value = os.path.join(path_to_dummy_data, urllib.parse.quote_plus(data_url.split('/')[-1]))
        if os.path.exists(value) or not self.load_existing_dummy_data:
            return value
        else:
            return path_to_dummy_data

    def delete_extracted_files(self):
        pass

    def manage_extracted_files(self):
        pass

    def iter_archive(self, path):

        def _iter_archive_members(path):
            dummy_parent_path = Path(self.dummy_file).parent
            relative_path = path.relative_to(dummy_parent_path)
            with ZipFile(self.local_path_to_dummy_data) as zip_file:
                members = zip_file.namelist()
            for member in members:
                if member.startswith(relative_path.as_posix()):
                    yield dummy_parent_path.joinpath(member)
        path = Path(path)
        file_paths = _iter_archive_members(path) if self.use_local_dummy_data else path.rglob('*')
        for file_path in file_paths:
            if file_path.is_file() and (not file_path.name.startswith(('.', '__'))):
                yield (file_path.relative_to(path).as_posix(), file_path.open('rb'))

    def iter_files(self, paths):
        if not isinstance(paths, list):
            paths = [paths]
        for path in paths:
            if os.path.isfile(path):
                yield path
            else:
                for dirpath, dirnames, filenames in os.walk(path):
                    if os.path.basename(dirpath).startswith(('.', '__')):
                        continue
                    dirnames.sort()
                    for filename in sorted(filenames):
                        if filename.startswith(('.', '__')):
                            continue
                        yield os.path.join(dirpath, filename)