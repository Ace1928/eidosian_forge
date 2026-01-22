import importlib
import json
import time
import datetime
import os
import requests
import shutil
import hashlib
import tqdm
import math
import zipfile
import parlai.utils.logging as logging
class DownloadableFile:
    """
    A class used to abstract any file that has to be downloaded online.

    Any task that needs to download a file needs to have a list RESOURCES
    that have objects of this class as elements.

    This class provides the following functionality:

    - Download a file from a URL / Google Drive
    - Untar the file if zipped
    - Checksum for the downloaded file
    - Send HEAD request to validate URL or Google Drive link

    An object of this class needs to be created with:

    - url <string> : URL or Google Drive id to download from
    - file_name <string> : File name that the file should be named
    - hashcode <string> : SHA256 hashcode of the downloaded file
    - zipped <boolean> : False if the file is not compressed
    - from_google <boolean> : True if the file is from Google Drive
    """

    def __init__(self, url, file_name, hashcode, zipped=True, from_google=False):
        self.url = url
        self.file_name = file_name
        self.hashcode = hashcode
        self.zipped = zipped
        self.from_google = from_google

    def checksum(self, dpath):
        """
        Checksum on a given file.

        :param dpath: path to the downloaded file.
        """
        sha256_hash = hashlib.sha256()
        with open(os.path.join(dpath, self.file_name), 'rb') as f:
            for byte_block in iter(lambda: f.read(65536), b''):
                sha256_hash.update(byte_block)
            if sha256_hash.hexdigest() != self.hashcode:
                raise AssertionError(f'[ Checksum for {self.file_name} from \n{self.url}\ndoes not match the expected checksum. Please try again. ]')
            else:
                logging.debug('Checksum Successful')

    def download_file(self, dpath):
        if self.from_google:
            download_from_google_drive(self.url, os.path.join(dpath, self.file_name))
        else:
            download(self.url, dpath, self.file_name)
        self.checksum(dpath)
        if self.zipped:
            untar(dpath, self.file_name)

    def check_header(self):
        """
        Performs a HEAD request to check if the URL / Google Drive ID is live.
        """
        session = requests.Session()
        if self.from_google:
            URL = 'https://docs.google.com/uc?export=download'
            response = session.head(URL, params={'id': self.url}, stream=True)
        else:
            headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36'}
            response = session.head(self.url, allow_redirects=True, headers=headers)
        status = response.status_code
        session.close()
        assert status == 200