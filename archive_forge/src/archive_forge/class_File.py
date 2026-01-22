import sys
import os
from warnings import warn
from .compat import filename_decode, filename_encode
from .base import phil, with_phil
from .group import Group
from .. import h5, h5f, h5p, h5i, h5fd, _objects
from .. import version
class File(Group):
    """
        Represents an HDF5 file.
    """

    @property
    def attrs(self):
        """ Attributes attached to this object """
        from . import attrs
        with phil:
            return attrs.AttributeManager(self['/'])

    @property
    @with_phil
    def filename(self):
        """File name on disk"""
        return filename_decode(h5f.get_name(self.id))

    @property
    @with_phil
    def driver(self):
        """Low-level HDF5 file driver used to open file"""
        drivers = {h5fd.SEC2: 'sec2', h5fd.STDIO: 'stdio', h5fd.CORE: 'core', h5fd.FAMILY: 'family', h5fd.WINDOWS: 'windows', h5fd.MPIO: 'mpio', h5fd.MPIPOSIX: 'mpiposix', h5fd.fileobj_driver: 'fileobj'}
        if ros3:
            drivers[h5fd.ROS3D] = 'ros3'
        if direct_vfd:
            drivers[h5fd.DIRECT] = 'direct'
        return drivers.get(self.id.get_access_plist().get_driver(), 'unknown')

    @property
    @with_phil
    def mode(self):
        """ Python mode used to open file """
        write_intent = h5f.ACC_RDWR
        if swmr_support:
            write_intent |= h5f.ACC_SWMR_WRITE
        return 'r+' if self.id.get_intent() & write_intent else 'r'

    @property
    @with_phil
    def libver(self):
        """File format version bounds (2-tuple: low, high)"""
        bounds = self.id.get_access_plist().get_libver_bounds()
        return tuple((libver_dict_r[x] for x in bounds))

    @property
    @with_phil
    def userblock_size(self):
        """ User block size (in bytes) """
        fcpl = self.id.get_create_plist()
        return fcpl.get_userblock()

    @property
    @with_phil
    def meta_block_size(self):
        """ Meta block size (in bytes) """
        fapl = self.id.get_access_plist()
        return fapl.get_meta_block_size()
    if mpi:

        @property
        @with_phil
        def atomic(self):
            """ Set/get MPI-IO atomic mode
            """
            return self.id.get_mpi_atomicity()

        @atomic.setter
        @with_phil
        def atomic(self, value):
            self.id.set_mpi_atomicity(value)

    @property
    @with_phil
    def swmr_mode(self):
        """ Controls single-writer multiple-reader mode """
        return swmr_support and bool(self.id.get_intent() & (h5f.ACC_SWMR_READ | h5f.ACC_SWMR_WRITE))

    @swmr_mode.setter
    @with_phil
    def swmr_mode(self, value):
        if value:
            self.id.start_swmr_write()
        else:
            raise ValueError('It is not possible to forcibly switch SWMR mode off.')

    def __init__(self, name, mode='r', driver=None, libver=None, userblock_size=None, swmr=False, rdcc_nslots=None, rdcc_nbytes=None, rdcc_w0=None, track_order=None, fs_strategy=None, fs_persist=False, fs_threshold=1, fs_page_size=None, page_buf_size=None, min_meta_keep=0, min_raw_keep=0, locking=None, alignment_threshold=1, alignment_interval=1, meta_block_size=None, **kwds):
        """Create a new file object.

        See the h5py user guide for a detailed explanation of the options.

        name
            Name of the file on disk, or file-like object.  Note: for files
            created with the 'core' driver, HDF5 still requires this be
            non-empty.
        mode
            r        Readonly, file must exist (default)
            r+       Read/write, file must exist
            w        Create file, truncate if exists
            w- or x  Create file, fail if exists
            a        Read/write if exists, create otherwise
        driver
            Name of the driver to use.  Legal values are None (default,
            recommended), 'core', 'sec2', 'direct', 'stdio', 'mpio', 'ros3'.
        libver
            Library version bounds.  Supported values: 'earliest', 'v108',
            'v110', 'v112'  and 'latest'. The 'v108', 'v110' and 'v112'
            options can only be specified with the HDF5 1.10.2 library or later.
        userblock_size
            Desired size of user block.  Only allowed when creating a new
            file (mode w, w- or x).
        swmr
            Open the file in SWMR read mode. Only used when mode = 'r'.
        rdcc_nbytes
            Total size of the dataset chunk cache in bytes. The default size
            is 1024**2 (1 MiB) per dataset. Applies to all datasets unless individually changed.
        rdcc_w0
            The chunk preemption policy for all datasets.  This must be
            between 0 and 1 inclusive and indicates the weighting according to
            which chunks which have been fully read or written are penalized
            when determining which chunks to flush from cache.  A value of 0
            means fully read or written chunks are treated no differently than
            other chunks (the preemption is strictly LRU) while a value of 1
            means fully read or written chunks are always preempted before
            other chunks.  If your application only reads or writes data once,
            this can be safely set to 1.  Otherwise, this should be set lower
            depending on how often you re-read or re-write the same data.  The
            default value is 0.75. Applies to all datasets unless individually changed.
        rdcc_nslots
            The number of chunk slots in the raw data chunk cache for this
            file. Increasing this value reduces the number of cache collisions,
            but slightly increases the memory used. Due to the hashing
            strategy, this value should ideally be a prime number. As a rule of
            thumb, this value should be at least 10 times the number of chunks
            that can fit in rdcc_nbytes bytes. For maximum performance, this
            value should be set approximately 100 times that number of
            chunks. The default value is 521. Applies to all datasets unless individually changed.
        track_order
            Track dataset/group/attribute creation order under root group
            if True. If None use global default h5.get_config().track_order.
        fs_strategy
            The file space handling strategy to be used.  Only allowed when
            creating a new file (mode w, w- or x).  Defined as:
            "fsm"        FSM, Aggregators, VFD
            "page"       Paged FSM, VFD
            "aggregate"  Aggregators, VFD
            "none"       VFD
            If None use HDF5 defaults.
        fs_page_size
            File space page size in bytes. Only used when fs_strategy="page". If
            None use the HDF5 default (4096 bytes).
        fs_persist
            A boolean value to indicate whether free space should be persistent
            or not.  Only allowed when creating a new file.  The default value
            is False.
        fs_threshold
            The smallest free-space section size that the free space manager
            will track.  Only allowed when creating a new file.  The default
            value is 1.
        page_buf_size
            Page buffer size in bytes. Only allowed for HDF5 files created with
            fs_strategy="page". Must be a power of two value and greater or
            equal than the file space page size when creating the file. It is
            not used by default.
        min_meta_keep
            Minimum percentage of metadata to keep in the page buffer before
            allowing pages containing metadata to be evicted. Applicable only if
            page_buf_size is set. Default value is zero.
        min_raw_keep
            Minimum percentage of raw data to keep in the page buffer before
            allowing pages containing raw data to be evicted. Applicable only if
            page_buf_size is set. Default value is zero.
        locking
            The file locking behavior. Defined as:

            - False (or "false") --  Disable file locking
            - True (or "true")   --  Enable file locking
            - "best-effort"      --  Enable file locking but ignore some errors
            - None               --  Use HDF5 defaults

            .. warning::

                The HDF5_USE_FILE_LOCKING environment variable can override
                this parameter.

            Only available with HDF5 >= 1.12.1 or 1.10.x >= 1.10.7.

        alignment_threshold
            Together with ``alignment_interval``, this property ensures that
            any file object greater than or equal in size to the alignement
            threshold (in bytes) will be aligned on an address which is a
            multiple of alignment interval.

        alignment_interval
            This property should be used in conjunction with
            ``alignment_threshold``. See the description above. For more
            details, see
            https://portal.hdfgroup.org/display/HDF5/H5P_SET_ALIGNMENT

        meta_block_size
            Set the current minimum size, in bytes, of new metadata block allocations.
            See https://portal.hdfgroup.org/display/HDF5/H5P_SET_META_BLOCK_SIZE

        Additional keywords
            Passed on to the selected file driver.
        """
        if driver == 'ros3':
            if ros3:
                from urllib.parse import urlparse
                url = urlparse(name)
                if url.scheme == 's3':
                    aws_region = kwds.get('aws_region', b'').decode('ascii')
                    if len(aws_region) == 0:
                        raise ValueError('AWS region required for s3:// location')
                    name = f'https://s3.{aws_region}.amazonaws.com/{url.netloc}{url.path}'
                elif url.scheme not in ('https', 'http'):
                    raise ValueError(f'{name}: S3 location must begin with either "https://", "http://", or "s3://"')
            else:
                raise ValueError("h5py was built without ROS3 support, can't use ros3 driver")
        if locking is not None and hdf5_version < (1, 12, 1) and (hdf5_version[:2] != (1, 10) or hdf5_version[2] < 7):
            raise ValueError('HDF5 version >= 1.12.1 or 1.10.x >= 1.10.7 required for file locking options.')
        if isinstance(name, _objects.ObjectID):
            if fs_strategy:
                raise ValueError('Unable to set file space strategy of an existing file')
            with phil:
                fid = h5i.get_file_id(name)
        else:
            if hasattr(name, 'read') and hasattr(name, 'seek'):
                if driver not in (None, 'fileobj'):
                    raise ValueError("Driver must be 'fileobj' for file-like object if specified.")
                driver = 'fileobj'
                if kwds.get('fileobj', name) != name:
                    raise ValueError("Invalid value of 'fileobj' argument; must equal to file-like object if specified.")
                kwds.update(fileobj=name)
                name = repr(name).encode('ASCII', 'replace')
            else:
                name = filename_encode(name)
            if track_order is None:
                track_order = h5.get_config().track_order
            if fs_strategy and mode not in ('w', 'w-', 'x'):
                raise ValueError('Unable to set file space strategy of an existing file')
            if swmr and mode != 'r':
                warn("swmr=True only affects read ('r') mode. For swmr write mode, set f.swmr_mode = True after opening the file.", stacklevel=2)
            with phil:
                fapl = make_fapl(driver, libver, rdcc_nslots, rdcc_nbytes, rdcc_w0, locking, page_buf_size, min_meta_keep, min_raw_keep, alignment_threshold=alignment_threshold, alignment_interval=alignment_interval, meta_block_size=meta_block_size, **kwds)
                fcpl = make_fcpl(track_order=track_order, fs_strategy=fs_strategy, fs_persist=fs_persist, fs_threshold=fs_threshold, fs_page_size=fs_page_size)
                fid = make_fid(name, mode, userblock_size, fapl, fcpl, swmr=swmr)
            if isinstance(libver, tuple):
                self._libver = libver
            else:
                self._libver = (libver, 'latest')
        super().__init__(fid)

    def close(self):
        """ Close the file.  All open objects become invalid """
        with phil:
            if self.id.valid:
                self.id._close_open_objects(h5f.OBJ_LOCAL | ~h5f.OBJ_FILE)
                self.id._close_open_objects(h5f.OBJ_LOCAL | h5f.OBJ_FILE)
                self.id.close()
                _objects.nonlocal_close()

    def flush(self):
        """ Tell the HDF5 library to flush its buffers.
        """
        with phil:
            h5f.flush(self.id)

    @with_phil
    def __enter__(self):
        return self

    @with_phil
    def __exit__(self, *args):
        if self.id:
            self.close()

    @with_phil
    def __repr__(self):
        if not self.id:
            r = '<Closed HDF5 file>'
        else:
            filename = self.filename
            if isinstance(filename, bytes):
                filename = filename.decode('utf8', 'replace')
            r = f'<HDF5 file "{os.path.basename(filename)}" (mode {self.mode})>'
        return r