import os
import sys
import ftplib
import warnings
from .utils import parse_url
class DOIDownloader:
    """
    Download manager for fetching files from Digital Object Identifiers (DOIs).

    Open-access data repositories often issue Digital Object Identifiers (DOIs)
    for data which provide a stable link and citation point. The trick is
    finding out the download URL for a file given the DOI.

    When called, this downloader uses the repository's public API to find out
    the download URL from the DOI and file name. It then uses
    :class:`pooch.HTTPDownloader` to download the URL into the specified local
    file. Allowing "URL"s  to be specified with the DOI instead of the actual
    HTTP download link. Uses the :mod:`requests` library to manage downloads
    and interact with the APIs.

    The **format of the "URL"** is: ``doi:{DOI}/{file name}``.

    Notice that there are no ``//`` like in HTTP/FTP and you must specify a
    file name after the DOI (separated by a ``/``).

    Use with :meth:`pooch.Pooch.fetch` or :func:`pooch.retrieve` to be able to
    download files given the DOI instead of an HTTP link.

    Supported repositories:

    * `figshare <https://www.figshare.com>`__
    * `Zenodo <https://www.zenodo.org>`__
    * `Dataverse <https://dataverse.org/>`__ instances

    .. attention::

        DOIs from other repositories **will not work** since we need to access
        their particular APIs to find the download links. We welcome
        suggestions and contributions adding new repositories.

    Parameters
    ----------
    progressbar : bool or an arbitrary progress bar object
        If True, will print a progress bar of the download to standard error
        (stderr). Requires `tqdm <https://github.com/tqdm/tqdm>`__ to be
        installed. Alternatively, an arbitrary progress bar object can be
        passed. See :ref:`custom-progressbar` for details.
    chunk_size : int
        Files are streamed *chunk_size* bytes at a time instead of loading
        everything into memory at one. Usually doesn't need to be changed.
    **kwargs
        All keyword arguments given when creating an instance of this class
        will be passed to :func:`requests.get`.

    Examples
    --------

    Download one of the data files from the figshare archive of Pooch test
    data:

    >>> import os
    >>> downloader = DOIDownloader()
    >>> url = "doi:10.6084/m9.figshare.14763051.v1/tiny-data.txt"
    >>> # Not using with Pooch.fetch so no need to pass an instance of Pooch
    >>> downloader(url=url, output_file="tiny-data.txt", pooch=None)
    >>> os.path.exists("tiny-data.txt")
    True
    >>> with open("tiny-data.txt") as f:
    ...     print(f.read().strip())
    # A tiny data file for test purposes only
    1  2  3  4  5  6
    >>> os.remove("tiny-data.txt")

    Same thing but for our Zenodo archive:

    >>> url = "doi:10.5281/zenodo.4924875/tiny-data.txt"
    >>> downloader(url=url, output_file="tiny-data.txt", pooch=None)
    >>> os.path.exists("tiny-data.txt")
    True
    >>> with open("tiny-data.txt") as f:
    ...     print(f.read().strip())
    # A tiny data file for test purposes only
    1  2  3  4  5  6
    >>> os.remove("tiny-data.txt")

    """

    def __init__(self, progressbar=False, chunk_size=1024, **kwargs):
        self.kwargs = kwargs
        self.progressbar = progressbar
        self.chunk_size = chunk_size

    def __call__(self, url, output_file, pooch):
        """
        Download the given DOI URL over HTTP to the given output file.

        Uses the repository's API to determine the actual HTTP download URL
        from the given DOI.

        Uses :func:`requests.get`.

        Parameters
        ----------
        url : str
            The URL to the file you want to download.
        output_file : str or file-like object
            Path (and file name) to which the file will be downloaded.
        pooch : :class:`~pooch.Pooch`
            The instance of :class:`~pooch.Pooch` that is calling this method.

        """
        parsed_url = parse_url(url)
        data_repository = doi_to_repository(parsed_url['netloc'])
        file_name = parsed_url['path']
        if file_name[0] == '/':
            file_name = file_name[1:]
        download_url = data_repository.download_url(file_name)
        downloader = HTTPDownloader(progressbar=self.progressbar, chunk_size=self.chunk_size, **self.kwargs)
        downloader(download_url, output_file, pooch)