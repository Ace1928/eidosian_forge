import warnings
from io import BytesIO
import numpy as np
from . import analyze  # module import
from .batteryrunners import Report
from .optpkg import optional_package
from .spatialimages import HeaderDataError, HeaderTypeError
class Spm99AnalyzeImage(analyze.AnalyzeImage):
    """Class for SPM99 variant of basic Analyze image"""
    header_class = Spm99AnalyzeHeader
    header: Spm99AnalyzeHeader
    files_types = (('image', '.img'), ('header', '.hdr'), ('mat', '.mat'))
    has_affine = True
    makeable = True
    rw = have_scipy

    @classmethod
    def from_file_map(klass, file_map, *, mmap=True, keep_file_open=None):
        """Class method to create image from mapping in ``file_map``

        Parameters
        ----------
        file_map : dict
            Mapping with (kay, value) pairs of (``file_type``, FileHolder
            instance giving file-likes for each file needed for this image
            type.
        mmap : {True, False, 'c', 'r'}, optional, keyword only
            `mmap` controls the use of numpy memory mapping for reading image
            array data.  If False, do not try numpy ``memmap`` for data array.
            If one of {'c', 'r'}, try numpy memmap with ``mode=mmap``.  A
            `mmap` value of True gives the same behavior as ``mmap='c'``.  If
            image data file cannot be memory-mapped, ignore `mmap` value and
            read array from file.
        keep_file_open : { None, True, False }, optional, keyword only
            `keep_file_open` controls whether a new file handle is created
            every time the image is accessed, or a single file handle is
            created and used for the lifetime of this ``ArrayProxy``. If
            ``True``, a single file handle is created and used. If ``False``,
            a new file handle is created every time the image is accessed.
            If ``file_map`` refers to an open file handle, this setting has no
            effect. The default value (``None``) will result in the value of
            ``nibabel.arrayproxy.KEEP_FILE_OPEN_DEFAULT`` being used.

        Returns
        -------
        img : Spm99AnalyzeImage instance

        """
        ret = super().from_file_map(file_map, mmap=mmap, keep_file_open=keep_file_open)
        try:
            matf = file_map['mat'].get_prepare_fileobj()
        except OSError:
            return ret
        with matf:
            contents = matf.read()
        if len(contents) == 0:
            return ret
        import scipy.io as sio
        mats = sio.loadmat(BytesIO(contents))
        if 'mat' in mats:
            mat = mats['mat']
            if mat.ndim > 2:
                warnings.warn('More than one affine in "mat" matrix, using first')
                mat = mat[:, :, 0]
            ret._affine = mat
        elif 'M' in mats:
            hdr = ret._header
            if hdr.default_x_flip:
                ret._affine = np.dot(np.diag([-1, 1, 1, 1]), mats['M'])
            else:
                ret._affine = mats['M']
        else:
            raise ValueError('mat file found but no "mat" or "M" in it')
        to_111 = np.eye(4)
        to_111[:3, 3] = 1
        ret._affine = np.dot(ret._affine, to_111)
        return ret

    def to_file_map(self, file_map=None, dtype=None):
        """Write image to `file_map` or contained ``self.file_map``

        Extends Analyze ``to_file_map`` method by writing ``mat`` file

        Parameters
        ----------
        file_map : None or mapping, optional
           files mapping.  If None (default) use object's ``file_map``
           attribute instead
        """
        if file_map is None:
            file_map = self.file_map
        super().to_file_map(file_map, dtype=dtype)
        mat = self._affine
        if mat is None:
            return
        import scipy.io as sio
        hdr = self._header
        if hdr.default_x_flip:
            M = np.dot(np.diag([-1, 1, 1, 1]), mat)
        else:
            M = mat
        from_111 = np.eye(4)
        from_111[:3, 3] = -1
        M = np.dot(M, from_111)
        mat = np.dot(mat, from_111)
        with file_map['mat'].get_prepare_fileobj(mode='wb') as mfobj:
            sio.savemat(mfobj, {'M': M, 'mat': mat}, format='4')