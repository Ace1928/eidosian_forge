import bz2
import gzip
import os
import platform
import numpy as np
class UrlCached(object):

    def __init__(self, url):
        self.url = url
        self.path = os.path.join(data_dir, os.path.split(url)[1])

    def download(self, force=False):
        if not os.path.exists(self.path) or force:
            print('Downloading %s to %s' % (self.url, self.path))
            code = os.system(self.download_command_wget())
            if not os.path.exists(self.path):
                print('Download failed, exit code was: ' + str(code) + ' will try with curl')
                code = os.system(self.download_command_curl())
                if not os.path.exists(self.path):
                    print('Download failed again, exit code was: ' + str(code) + ' using urlretrieve')
                    self.download_urlretrieve()

    def fetch(self):
        self.download()
        if os.path.exists(self.path):
            return self.path
        else:
            raise Exception('file not found and/or download failed')

    def download_command_wget(self):
        return 'wget --progress=bar:force -c -P %s %s' % (data_dir, self.url)

    def download_command_curl(self):
        return 'cd %s; curl -O -L %s' % (data_dir, self.url)

    def download_urlretrieve(self):
        urlretrieve(self.url, self.path)