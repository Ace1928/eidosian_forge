from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import re
import stat
import tarfile
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import local_file_adapter
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import exceptions as creds_exceptions
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import http_encoding
from googlecloudsdk.core.util import retry
import requests
import six
def DownloadAndExtractTar(url, download_dir, extract_dir, progress_callback=None, command_path='unknown'):
    """Download and extract the given tar file.

  Args:
    url: str, The URL to download.
    download_dir: str, The path to put the temporary download file into.
    extract_dir: str, The path to extract the tar into.
    progress_callback: f(float), A function to call with the fraction of
      completeness.
    command_path: the command path to include in the User-Agent header if the
      URL is HTTP

  Returns:
    [str], The files that were extracted from the tar file.

  Raises:
    URLFetchError: If there is a problem fetching the given URL.
  """
    for d in [download_dir, extract_dir]:
        if not os.path.exists(d):
            file_utils.MakeDir(d)
    download_file_path = os.path.join(download_dir, os.path.basename(url))
    if os.path.exists(download_file_path):
        os.remove(download_file_path)
    download_callback, install_callback = console_io.SplitProgressBar(progress_callback, [1, 1])
    try:
        response = MakeRequest(url, command_path)
        with file_utils.BinaryFileWriter(download_file_path) as fp:
            total_written = 0
            total_size = len(response.content)
            for chunk in response.iter_content(chunk_size=WRITE_BUFFER_SIZE):
                fp.write(chunk)
                total_written += len(chunk)
                download_callback(total_written / total_size)
        download_callback(1)
    except (requests.exceptions.HTTPError, OSError) as e:
        raise URLFetchError(e)
    with tarfile.open(name=download_file_path) as tar:
        members = tar.getmembers()
        total_files = len(members)
        files = []
        for num, member in enumerate(members, start=1):
            files.append(member.name + '/' if member.isdir() else member.name)
            tar.extract(member, extract_dir)
            full_path = os.path.join(extract_dir, member.name)
            if os.path.isfile(full_path) and (not os.access(full_path, os.W_OK)):
                os.chmod(full_path, stat.S_IWUSR | stat.S_IREAD)
            install_callback(num / total_files)
        install_callback(1)
    os.remove(download_file_path)
    return files