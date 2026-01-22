from math import log
import os
from os import path as op
import sys
import shutil
import time
from . import appdata_dir, resource_dirs
from . import StdoutProgressIndicator, urlopen
def _fetch_file(url, file_name, print_destination=True):
    """Load requested file, downloading it if needed or requested

    Parameters
    ----------
    url: string
        The url of file to be downloaded.
    file_name: string
        Name, along with the path, of where downloaded file will be saved.
    print_destination: bool, optional
        If true, destination of where file was saved will be printed after
        download finishes.
    resume: bool, optional
        If true, try to resume partially downloaded files.
    """
    print('Imageio: %r was not found on your computer; downloading it now.' % os.path.basename(file_name))
    temp_file_name = file_name + '.part'
    local_file = None
    initial_size = 0
    errors = []
    for tries in range(4):
        try:
            remote_file = urlopen(url, timeout=5.0)
            file_size = int(remote_file.headers['Content-Length'].strip())
            size_str = _sizeof_fmt(file_size)
            print('Try %i. Download from %s (%s)' % (tries + 1, url, size_str))
            local_file = open(temp_file_name, 'wb')
            _chunk_read(remote_file, local_file, initial_size=initial_size)
            if not local_file.closed:
                local_file.close()
            shutil.move(temp_file_name, file_name)
            if print_destination is True:
                sys.stdout.write('File saved as %s.\n' % file_name)
            break
        except Exception as e:
            errors.append(e)
            print('Error while fetching file: %s.' % str(e))
        finally:
            if local_file is not None:
                if not local_file.closed:
                    local_file.close()
    else:
        raise IOError('Unable to download %r. Perhaps there is no internet connection? If there is, please report this problem.' % os.path.basename(file_name))