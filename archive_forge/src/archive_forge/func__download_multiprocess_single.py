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
def _download_multiprocess_single(url, path, dest_fname):
    """
    Helper function to download an individual item.

    Unlike download() above, does not deal with downloading chunks of a big
    file, does not support retries (and does not fail if retries are exhausted).

    :param url: URL to download from
    :param path: directory to save in
    :param dest_fname: destination file name of image
    :return tuple (dest_fname, http status)
    """
    status = None
    error_msg = None
    try:
        headers = {}
        response = requests.get(url, stream=False, timeout=10, allow_redirects=True, headers=headers)
    except Exception as e:
        status = 500
        error_msg = '[Exception during download during fetching] ' + str(e)
        return (dest_fname, status, error_msg)
    if response.ok:
        try:
            with open(os.path.join(path, dest_fname), 'wb+') as out_file:
                response.raw.decode_content = True
                out_file.write(response.content)
            status = 200
        except Exception as e:
            status = 500
            error_msg = '[Exception during decoding or writing] ' + str(e)
    else:
        status = response.status_code
        error_msg = '[Response not OK] Response: %s' % response
    return (dest_fname, status, error_msg)