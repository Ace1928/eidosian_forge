import abc
import os
import bz2
import gzip
import lzma
import shutil
from zipfile import ZipFile
from tarfile import TarFile
from .utils import get_logger
class Untar(ExtractorProcessor):
    """
    Processor that unpacks a tar archive and returns a list of all files.

    Use with :meth:`pooch.Pooch.fetch` or :func:`pooch.retrieve` to untar a
    downloaded data file into a folder in the local data store. The
    method/function will return a list with the names of the extracted files
    instead of the archive.

    The output folder is ``{fname}.untar``.


    Parameters
    ----------
    members : list or None
        If None, will unpack all files in the archive. Otherwise, *members*
        must be a list of file names to unpack from the archive. Only these
        files will be unpacked.
    extract_dir : str or None
        If None, files will be unpacked to the default location (a folder in
        the same location as the downloaded tar file, with the suffix
        ``.untar`` added). Otherwise, files will be unpacked to
        ``extract_dir``, which is interpreted as a *relative path* (relative to
        the cache location  provided by :func:`pooch.retrieve` or
        :meth:`pooch.Pooch.fetch`).
    """

    @property
    def suffix(self):
        """
        String appended to unpacked archive folder name.
        Only used if extract_dir is None.
        """
        return '.untar'

    def _all_members(self, fname):
        """Return all members from a given archive."""
        with TarFile.open(fname, 'r') as tar_file:
            return [info.name for info in tar_file.getmembers()]

    def _extract_file(self, fname, extract_dir):
        """
        This method receives an argument for the archive to extract and the
        destination path.
        """
        with TarFile.open(fname, 'r') as tar_file:
            if self.members is None:
                get_logger().info("Untarring contents of '%s' to '%s'", fname, extract_dir)
                tar_file.extractall(path=extract_dir)
            else:
                for member in self.members:
                    get_logger().info("Extracting '%s' from '%s' to '%s'", member, fname, extract_dir)
                    subdir_members = [info for info in tar_file.getmembers() if os.path.normpath(info.name).startswith(os.path.normpath(member))]
                    tar_file.extractall(members=subdir_members, path=extract_dir)