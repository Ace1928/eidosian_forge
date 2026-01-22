import os
import string
import struct
import warnings
import numpy as np
import nibabel as nib
from nibabel.openers import Opener
from nibabel.orientations import aff2axcodes, axcodes2ornt
from nibabel.volumeutils import endian_codes, native_code, swapped_code
from .array_sequence import create_arraysequences_from_generator
from .header import Field
from .tractogram import LazyTractogram, Tractogram, TractogramItem
from .tractogram_file import DataError, HeaderError, HeaderWarning, TractogramFile
from .utils import peek_next
class TrkFile(TractogramFile):
    """Convenience class to encapsulate TRK file format.

    Notes
    -----
    TrackVis (so its file format: TRK) considers the streamline coordinate
    (0,0,0) to be in the corner of the voxel whereas NiBabel's streamlines
    internal representation (Voxel space) assumes (0,0,0) to be in the
    center of the voxel.

    Thus, streamlines are shifted by half a voxel on load and are shifted
    back on save.
    """
    MAGIC_NUMBER = b'TRACK'
    HEADER_SIZE = 1000
    SUPPORTS_DATA_PER_POINT = True
    SUPPORTS_DATA_PER_STREAMLINE = True

    def __init__(self, tractogram, header=None):
        """
        Parameters
        ----------
        tractogram : :class:`Tractogram` object
            Tractogram that will be contained in this :class:`TrkFile`.

        header : dict, optional
            Metadata associated to this tractogram file.

        Notes
        -----
        Streamlines of the tractogram are assumed to be in *RAS+*
        and *mm* space where coordinate (0,0,0) refers to the center
        of the voxel.
        """
        super().__init__(tractogram, header)

    @classmethod
    def is_correct_format(cls, fileobj):
        """Check if the file is in TRK format.

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object
            pointing to TRK file (and ready to read from the beginning
            of the TRK header data). Note that calling this function
            does not change the file position.

        Returns
        -------
        is_correct_format : {True, False}
            Returns True if `fileobj` is compatible with TRK format,
            otherwise returns False.
        """
        with Opener(fileobj) as f:
            magic_len = len(cls.MAGIC_NUMBER)
            magic_number = f.read(magic_len)
            f.seek(-magic_len, os.SEEK_CUR)
            return magic_number == cls.MAGIC_NUMBER

    @classmethod
    def _default_structarr(cls, endianness=None):
        """Return an empty compliant TRK header as numpy structured array"""
        dt = header_2_dtype
        if endianness is not None:
            endianness = endian_codes[endianness]
            dt = dt.newbyteorder(endianness)
        st_arr = np.zeros((), dtype=dt)
        st_arr[Field.MAGIC_NUMBER] = cls.MAGIC_NUMBER
        st_arr[Field.VOXEL_SIZES] = np.array((1, 1, 1), dtype='f4')
        st_arr[Field.DIMENSIONS] = np.array((1, 1, 1), dtype='h')
        st_arr[Field.VOXEL_TO_RASMM] = np.eye(4, dtype='f4')
        st_arr[Field.VOXEL_ORDER] = b'RAS'
        st_arr['version'] = 2
        st_arr['hdr_size'] = cls.HEADER_SIZE
        return st_arr

    @classmethod
    def create_empty_header(cls, endianness=None):
        """Return an empty compliant TRK header as dict"""
        st_arr = cls._default_structarr(endianness)
        return dict(zip(st_arr.dtype.names, st_arr.tolist()))

    @classmethod
    def load(cls, fileobj, lazy_load=False):
        """Loads streamlines from a filename or file-like object.

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object
            pointing to TRK file (and ready to read from the beginning
            of the TRK header). Note that calling this function
            does not change the file position.
        lazy_load : {False, True}, optional
            If True, load streamlines in a lazy manner i.e. they will not be
            kept in memory. Otherwise, load all streamlines in memory.

        Returns
        -------
        trk_file : :class:`TrkFile` object
            Returns an object containing tractogram data and header
            information.

        Notes
        -----
        Streamlines of the returned tractogram are assumed to be in *RAS*
        and *mm* space where coordinate (0,0,0) refers to the center of the
        voxel.
        """
        hdr = cls._read_header(fileobj)
        data_per_point_slice = {}
        if hdr[Field.NB_SCALARS_PER_POINT] > 0:
            cpt = 0
            for scalar_field in hdr['scalar_name']:
                scalar_name, nb_scalars = decode_value_from_name(scalar_field)
                if nb_scalars == 0:
                    continue
                slice_obj = slice(cpt, cpt + nb_scalars)
                data_per_point_slice[scalar_name] = slice_obj
                cpt += nb_scalars
            if cpt < hdr[Field.NB_SCALARS_PER_POINT]:
                slice_obj = slice(cpt, hdr[Field.NB_SCALARS_PER_POINT])
                data_per_point_slice['scalars'] = slice_obj
        data_per_streamline_slice = {}
        if hdr[Field.NB_PROPERTIES_PER_STREAMLINE] > 0:
            cpt = 0
            for property_field in hdr['property_name']:
                results = decode_value_from_name(property_field)
                property_name, nb_properties = results
                if nb_properties == 0:
                    continue
                slice_obj = slice(cpt, cpt + nb_properties)
                data_per_streamline_slice[property_name] = slice_obj
                cpt += nb_properties
            if cpt < hdr[Field.NB_PROPERTIES_PER_STREAMLINE]:
                slice_obj = slice(cpt, hdr[Field.NB_PROPERTIES_PER_STREAMLINE])
                data_per_streamline_slice['properties'] = slice_obj
        if lazy_load:

            def _read():
                for pts, scals, props in cls._read(fileobj, hdr):
                    items = data_per_point_slice.items()
                    data_for_points = {k: scals[:, v] for k, v in items}
                    items = data_per_streamline_slice.items()
                    data_for_streamline = {k: props[v] for k, v in items}
                    yield TractogramItem(pts, data_for_streamline, data_for_points)
            tractogram = LazyTractogram.from_data_func(_read)
        else:
            with Opener(fileobj) as f:
                old_file_position = f.tell()
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(old_file_position, os.SEEK_SET)
            mbytes = size // (1024 * 1024)
            sizes = [mbytes, 4, 4]
            if hdr['nb_scalars_per_point'] > 0:
                sizes = [mbytes // 2, mbytes // 2, 4]
            trk_reader = cls._read(fileobj, hdr)
            arr_seqs = create_arraysequences_from_generator(trk_reader, n=3, buffer_sizes=sizes)
            streamlines, scalars, properties = arr_seqs
            properties = np.asarray(properties)
            tractogram = Tractogram(streamlines)
            for name, slice_ in data_per_point_slice.items():
                tractogram.data_per_point[name] = scalars[:, slice_]
            for name, slice_ in data_per_streamline_slice.items():
                tractogram.data_per_streamline[name] = properties[:, slice_]
        tractogram.affine_to_rasmm = get_affine_trackvis_to_rasmm(hdr)
        tractogram = tractogram.to_world()
        return cls(tractogram, header=hdr)

    def save(self, fileobj):
        """Save tractogram to a filename or file-like object using TRK format.

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object
            pointing to TRK file (and ready to write from the beginning
            of the TRK header data).
        """
        header = self._default_structarr(endianness='little')
        for k, v in self.header.items():
            if k in header_2_dtype.fields.keys():
                header[k] = v
        if header[Field.VOXEL_ORDER] == b'':
            header[Field.VOXEL_ORDER] = b'LPS'
        nb_streamlines = 0
        nb_points = 0
        nb_scalars = 0
        nb_properties = 0
        with Opener(fileobj, mode='wb') as f:
            beginning = f.tell()
            f.write(header.tobytes())
            i4_dtype = np.dtype('<i4')
            f4_dtype = np.dtype('<f4')
            tractogram = self.tractogram.to_world(lazy=True)
            affine_to_trackvis = get_affine_rasmm_to_trackvis(header)
            tractogram = tractogram.apply_affine(affine_to_trackvis, lazy=True)
            tractogram = iter(tractogram)
            try:
                first_item, tractogram = peek_next(tractogram)
            except StopIteration:
                header[Field.NB_STREAMLINES] = 0
                header[Field.NB_SCALARS_PER_POINT] = 0
                header[Field.NB_PROPERTIES_PER_STREAMLINE] = 0
                f.seek(beginning, os.SEEK_SET)
                f.write(header.tobytes())
                return
            data_for_streamline = first_item.data_for_streamline
            if len(data_for_streamline) > MAX_NB_NAMED_PROPERTIES_PER_STREAMLINE:
                msg = f"Can only store {MAX_NB_NAMED_SCALARS_PER_POINT} named data_per_streamline (also known as 'properties' in the TRK format)."
                raise ValueError(msg)
            data_for_streamline_keys = sorted(data_for_streamline.keys())
            property_name = np.zeros(MAX_NB_NAMED_PROPERTIES_PER_STREAMLINE, dtype='S20')
            for i, name in enumerate(data_for_streamline_keys):
                nb_values = data_for_streamline[name].shape[-1]
                property_name[i] = encode_value_in_name(nb_values, name)
            header['property_name'][:] = property_name
            data_for_points = first_item.data_for_points
            if len(data_for_points) > MAX_NB_NAMED_SCALARS_PER_POINT:
                msg = f"Can only store {MAX_NB_NAMED_SCALARS_PER_POINT} named data_per_point (also known as 'scalars' in the TRK format)."
                raise ValueError(msg)
            data_for_points_keys = sorted(data_for_points.keys())
            scalar_name = np.zeros(MAX_NB_NAMED_SCALARS_PER_POINT, dtype='S20')
            for i, name in enumerate(data_for_points_keys):
                nb_values = data_for_points[name].shape[-1]
                scalar_name[i] = encode_value_in_name(nb_values, name)
            header['scalar_name'][:] = scalar_name
            for t in tractogram:
                if any((len(d) != len(t.streamline) for d in t.data_for_points.values())):
                    raise DataError('Missing scalars for some points!')
                points = np.asarray(t.streamline)
                scalars = [np.asarray(t.data_for_points[k]) for k in data_for_points_keys]
                scalars = np.concatenate([np.ndarray((len(points), 0))] + scalars, axis=1)
                properties = [np.asarray(t.data_for_streamline[k]) for k in data_for_streamline_keys]
                properties = np.concatenate([np.array([])] + properties).astype(f4_dtype)
                data = struct.pack(i4_dtype.str[:-1], len(points))
                pts_scalars = np.concatenate([points, scalars], axis=1).astype(f4_dtype)
                data += pts_scalars.tobytes()
                data += properties.tobytes()
                f.write(data)
                nb_streamlines += 1
                nb_points += len(points)
                nb_scalars += scalars.size
                nb_properties += len(properties)
            nb_scalars_per_point = nb_scalars / nb_points
            nb_properties_per_streamline = nb_properties / nb_streamlines
            if nb_scalars_per_point != int(nb_scalars_per_point):
                msg = 'Nb. of scalars differs from one point to another!'
                raise DataError(msg)
            if nb_properties_per_streamline != int(nb_properties_per_streamline):
                msg = 'Nb. of properties differs from one streamline to another!'
                raise DataError(msg)
            header[Field.NB_STREAMLINES] = nb_streamlines
            header[Field.NB_SCALARS_PER_POINT] = nb_scalars_per_point
            header[Field.NB_PROPERTIES_PER_STREAMLINE] = nb_properties_per_streamline
            f.seek(beginning, os.SEEK_SET)
            f.write(header.tobytes())

    @staticmethod
    def _read_header(fileobj):
        """Reads a TRK header from a file.

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object
            pointing to TRK file (and ready to read from the beginning
            of the TRK header). Note that calling this function
            does not change the file position.

        Returns
        -------
        header : dict
            Metadata associated with this tractogram file.
        """
        start_position = fileobj.tell() if hasattr(fileobj, 'tell') else None
        with Opener(fileobj) as f:
            header_buf = bytearray(header_2_dtype.itemsize)
            f.readinto(header_buf)
            header_rec = np.frombuffer(buffer=header_buf, dtype=header_2_dtype)
            endianness = native_code
            if header_rec['hdr_size'] != TrkFile.HEADER_SIZE:
                endianness = swapped_code
                header_rec = header_rec.view(header_rec.dtype.newbyteorder())
                if header_rec['hdr_size'] != TrkFile.HEADER_SIZE:
                    msg = f'Invalid hdr_size: {header_rec['hdr_size']} instead of {TrkFile.HEADER_SIZE}'
                    raise HeaderError(msg)
            if header_rec['version'] == 1:
                header_rec[Field.VOXEL_TO_RASMM] = np.zeros((4, 4))
            elif header_rec['version'] == 3:
                warnings.warn('Parsing a TRK v3 file as v2. Some features may not be handled correctly.', HeaderWarning)
            elif header_rec['version'] in (2, 3):
                pass
            else:
                raise HeaderError('NiBabel only supports versions 1 and 2 of the Trackvis file format')
            header = dict(zip(header_rec.dtype.names, header_rec[0]))
            header[Field.ENDIANNESS] = endianness
            if header[Field.VOXEL_TO_RASMM][3][3] == 0:
                header[Field.VOXEL_TO_RASMM] = np.eye(4, dtype=np.float32)
                warnings.warn("Field 'vox_to_ras' in the TRK's header was not recorded. Will continue assuming it's the identity.", HeaderWarning)
            axcodes = aff2axcodes(header[Field.VOXEL_TO_RASMM])
            if None in axcodes:
                msg = f"The 'vox_to_ras' affine is invalid! Could not determine the axis directions from it.\n{header[Field.VOXEL_TO_RASMM]}"
                raise HeaderError(msg)
            if header[Field.VOXEL_ORDER] == b'':
                msg = "Voxel order is not specified, will assume 'LPS' since it is Trackvis software's default."
                warnings.warn(msg, HeaderWarning)
                header[Field.VOXEL_ORDER] = b'LPS'
            header['_offset_data'] = f.tell()
        if start_position is not None:
            fileobj.seek(start_position, os.SEEK_SET)
        return header

    @staticmethod
    def _read(fileobj, header):
        """Return generator that reads TRK data from `fileobj` given `header`

        Parameters
        ----------
        fileobj : string or file-like object
            If string, a filename; otherwise an open file-like object
            pointing to TRK file (and ready to read from the beginning
            of the TRK header). Note that calling this function
            does not change the file position.
        header : dict
            Metadata associated with this tractogram file.

        Yields
        ------
        data : tuple of ndarrays
            Length 3 tuple of streamline data of form (points, scalars,
            properties), where:

            * points: ndarray of shape (n_pts, 3)
            * scalars: ndarray of shape (n_pts, nb_scalars_per_point)
            * properties: ndarray of shape (nb_properties_per_point,)
        """
        i4_dtype = np.dtype(header[Field.ENDIANNESS] + 'i4')
        f4_dtype = np.dtype(header[Field.ENDIANNESS] + 'f4')
        with Opener(fileobj) as f:
            start_position = f.tell()
            nb_pts_and_scalars = int(3 + header[Field.NB_SCALARS_PER_POINT])
            pts_and_scalars_size = int(nb_pts_and_scalars * f4_dtype.itemsize)
            nb_properties = header[Field.NB_PROPERTIES_PER_STREAMLINE]
            properties_size = int(nb_properties * f4_dtype.itemsize)
            f.seek(header['_offset_data'], os.SEEK_SET)
            nb_streamlines = header[Field.NB_STREAMLINES]
            if nb_streamlines == 0:
                nb_streamlines = np.inf
            count = 0
            nb_pts_dtype = i4_dtype.str[:-1]
            while count < nb_streamlines:
                nb_pts_str = f.read(i4_dtype.itemsize)
                if len(nb_pts_str) == 0:
                    break
                nb_pts = struct.unpack(nb_pts_dtype, nb_pts_str)[0]
                points_and_scalars = np.ndarray(shape=(nb_pts, nb_pts_and_scalars), dtype=f4_dtype, buffer=f.read(nb_pts * pts_and_scalars_size))
                points = points_and_scalars[:, :3]
                scalars = points_and_scalars[:, 3:]
                properties = np.ndarray(shape=(nb_properties,), dtype=f4_dtype, buffer=f.read(properties_size))
                yield (points, scalars, properties)
                count += 1
            header[Field.NB_STREAMLINES] = count
            f.seek(start_position, os.SEEK_CUR)

    def __str__(self):
        """Gets a formatted string of the header of a TRK file.

        Returns
        -------
        info : string
            Header information relevant to the TRK format.
        """
        vars = self.header.copy()
        for attr in dir(Field):
            if attr[0] in string.ascii_uppercase:
                hdr_field = getattr(Field, attr)
                if hdr_field in vars:
                    vars[attr] = vars[hdr_field]
        nb_scalars = self.header[Field.NB_SCALARS_PER_POINT]
        scalar_names = [s.decode('latin-1') for s in vars['scalar_name'][:nb_scalars] if len(s) > 0]
        vars['scalar_names'] = '\n  '.join(scalar_names)
        nb_properties = self.header[Field.NB_PROPERTIES_PER_STREAMLINE]
        property_names = [s.decode('latin-1') for s in vars['property_name'][:nb_properties] if len(s) > 0]
        vars['property_names'] = '\n  '.join(property_names)
        vars = {k: v.decode('latin-1') if hasattr(v, 'decode') else v for k, v in vars.items()}
        return 'MAGIC NUMBER: {MAGIC_NUMBER}\nv.{version}\ndim: {DIMENSIONS}\nvoxel_sizes: {VOXEL_SIZES}\norigin: {ORIGIN}\nnb_scalars: {NB_SCALARS_PER_POINT}\nscalar_names:\n  {scalar_names}\nnb_properties: {NB_PROPERTIES_PER_STREAMLINE}\nproperty_names:\n  {property_names}\nvox_to_world:\n{VOXEL_TO_RASMM}\nvoxel_order: {VOXEL_ORDER}\nimage_orientation_patient: {image_orientation_patient}\npad1: {pad1}\npad2: {pad2}\ninvert_x: {invert_x}\ninvert_y: {invert_y}\ninvert_z: {invert_z}\nswap_xy: {swap_xy}\nswap_yz: {swap_yz}\nswap_zx: {swap_zx}\nn_count: {NB_STREAMLINES}\nhdr_size: {hdr_size}'.format(**vars)