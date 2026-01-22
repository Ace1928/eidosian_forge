import abc
import os
import bz2
import gzip
import lzma
import shutil
from zipfile import ZipFile
from tarfile import TarFile
from .utils import get_logger
class Unzip(ExtractorProcessor):
    """
    Processor that unpacks a zip archive and returns a list of all files.

    Use with :meth:`pooch.Pooch.fetch` or :func:`pooch.retrieve` to unzip a
    downloaded data file into a folder in the local data store. The
    method/function will return a list with the names of the unzipped files
    instead of the zip archive.

    The output folder is ``{fname}.unzip``.

    Parameters
    ----------
    members : list or None
        If None, will unpack all files in the zip archive. Otherwise, *members*
        must be a list of file names to unpack from the archive. Only these
        files will be unpacked.
    extract_dir : str or None
        If None, files will be unpacked to the default location (a folder in
        the same location as the downloaded zip file, with the suffix
        ``.unzip`` added). Otherwise, files will be unpacked to
        ``extract_dir``, which is interpreted as a *relative path* (relative to
        the cache location provided by :func:`pooch.retrieve` or
        :meth:`pooch.Pooch.fetch`).

    """

    @property
    def suffix(self):
        """
        String appended to unpacked archive folder name.
        Only used if extract_dir is None.
        """
        return '.unzip'

    def _all_members(self, fname):
        """Return all members from a given archive."""
        with ZipFile(fname, 'r') as zip_file:
            return zip_file.namelist()

    def _extract_file(self, fname, extract_dir):
        """
        This method receives an argument for the archive to extract and the
        destination path.
        """
        with ZipFile(fname, 'r') as zip_file:
            if self.members is None:
                get_logger().info("Unzipping contents of '%s' to '%s'", fname, extract_dir)
                zip_file.extractall(path=extract_dir)
            else:
                for member in self.members:
                    get_logger().info("Extracting '%s' from '%s' to '%s'", member, fname, extract_dir)
                    subdir_members = [name for name in zip_file.namelist() if os.path.normpath(name).startswith(os.path.normpath(member))]
                    zip_file.extractall(members=subdir_members, path=extract_dir)