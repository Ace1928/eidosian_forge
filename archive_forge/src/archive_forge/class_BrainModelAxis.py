from the :meth:`.cifti2.Cifti2Header.get_axis` method on the header object
import abc
from operator import xor
import numpy as np
from . import cifti2
class BrainModelAxis(Axis):
    """
    Each row/column in the CIFTI-2 vector/matrix represents a single vertex or voxel

    This Axis describes which vertex/voxel is represented by each row/column.
    """

    def __init__(self, name, voxel=None, vertex=None, affine=None, volume_shape=None, nvertices=None):
        """
        New BrainModelAxis axes can be constructed by passing on the greyordinate brain-structure
        names and voxel/vertex indices to the constructor or by one of the
        factory methods:

        - :py:meth:`~BrainModelAxis.from_mask`: creates surface or volumetric BrainModelAxis axis
          from respectively 1D or 3D masks
        - :py:meth:`~BrainModelAxis.from_surface`: creates a surface BrainModelAxis axis

        The resulting BrainModelAxis axes can be concatenated by adding them together.

        Parameters
        ----------
        name : array_like
            brain structure name or (N, ) string array with the brain structure names
        voxel : array_like, optional
            (N, 3) array with the voxel indices (can be omitted for CIFTI-2 files only
            covering the surface)
        vertex :  array_like, optional
            (N, ) array with the vertex indices (can be omitted for volumetric CIFTI-2 files)
        affine : array_like, optional
            (4, 4) array mapping voxel indices to mm space (not needed for CIFTI-2 files only
            covering the surface)
        volume_shape : tuple of three integers, optional
            shape of the volume in which the voxels were defined (not needed for CIFTI-2 files only
            covering the surface)
        nvertices : dict from string to integer, optional
            maps names of surface elements to integers (not needed for volumetric CIFTI-2 files)
        """
        if voxel is None:
            if vertex is None:
                raise ValueError('At least one of voxel or vertex indices should be defined')
            nelements = len(vertex)
            self.voxel = np.full((nelements, 3), fill_value=-1, dtype=int)
        else:
            nelements = len(voxel)
            self.voxel = np.asanyarray(voxel, dtype=int)
        if vertex is None:
            self.vertex = np.full(nelements, fill_value=-1, dtype=int)
        else:
            self.vertex = np.asanyarray(vertex, dtype=int)
        if isinstance(name, str):
            name = [self.to_cifti_brain_structure_name(name)] * self.vertex.size
        self.name = np.asanyarray(name, dtype='U')
        if nvertices is None:
            self.nvertices = {}
        else:
            self.nvertices = {self.to_cifti_brain_structure_name(name): number for name, number in nvertices.items()}
        for name in list(self.nvertices.keys()):
            if name not in self.name:
                del self.nvertices[name]
        surface_mask = self.surface_mask
        if surface_mask.all():
            self.affine = None
            self.volume_shape = None
        else:
            if affine is None or volume_shape is None:
                raise ValueError('Affine and volume shape should be defined for BrainModelAxis containing voxels')
            self.affine = np.asanyarray(affine)
            self.volume_shape = volume_shape
        if np.any(self.vertex[surface_mask] < 0):
            raise ValueError('Undefined vertex indices found for surface elements')
        if np.any(self.voxel[~surface_mask] < 0):
            raise ValueError('Undefined voxel indices found for volumetric elements')
        for check_name in ('name', 'voxel', 'vertex'):
            shape = (self.size, 3) if check_name == 'voxel' else (self.size,)
            if getattr(self, check_name).shape != shape:
                raise ValueError(f'Input {check_name} has incorrect shape ({getattr(self, check_name).shape}) for BrainModelAxis axis')

    @classmethod
    def from_mask(cls, mask, name='other', affine=None):
        """
        Creates a new BrainModelAxis axis describing the provided mask

        Parameters
        ----------
        mask : array_like
            all non-zero voxels will be included in the BrainModelAxis axis
            should be (Nx, Ny, Nz) array for volume mask or (Nvertex, ) array for surface mask
        name : str, optional
            Name of the brain structure (e.g. 'CortexRight', 'thalamus_left' or 'brain_stem')
        affine : array_like, optional
            (4, 4) array with the voxel to mm transformation (defaults to identity matrix)
            Argument will be ignored for surface masks

        Returns
        -------
        BrainModelAxis which covers the provided mask
        """
        if affine is None:
            affine = np.eye(4)
        else:
            affine = np.asanyarray(affine)
        if affine.shape != (4, 4):
            raise ValueError(f'Affine transformation should be a 4x4 array or None, not {affine!r}')
        mask = np.asanyarray(mask)
        if mask.ndim == 1:
            return cls.from_surface(np.where(mask != 0)[0], mask.size, name=name)
        elif mask.ndim == 3:
            voxels = np.array(np.where(mask != 0)).T
            return cls(name, voxel=voxels, affine=affine, volume_shape=mask.shape)
        else:
            raise ValueError('Mask should be either 1-dimensional (for surfaces) or 3-dimensional (for volumes), not %i-dimensional' % mask.ndim)

    @classmethod
    def from_surface(cls, vertices, nvertex, name='Other'):
        """
        Creates a new BrainModelAxis axis describing the vertices on a surface

        Parameters
        ----------
        vertices : array_like
            indices of the vertices on the surface
        nvertex : int
            total number of vertices on the surface
        name : str
            Name of the brain structure (e.g. 'CortexLeft' or 'CortexRight')

        Returns
        -------
        BrainModelAxis which covers (part of) the surface
        """
        cifti_name = cls.to_cifti_brain_structure_name(name)
        return cls(cifti_name, vertex=vertices, nvertices={cifti_name: nvertex})

    @classmethod
    def from_index_mapping(cls, mim):
        """
        Creates a new BrainModel axis based on a CIFTI-2 dataset

        Parameters
        ----------
        mim : :class:`.cifti2.Cifti2MatrixIndicesMap`

        Returns
        -------
        BrainModelAxis
        """
        nbm = sum((bm.index_count for bm in mim.brain_models))
        voxel = np.full((nbm, 3), fill_value=-1, dtype=int)
        vertex = np.full(nbm, fill_value=-1, dtype=int)
        name = []
        nvertices = {}
        affine, shape = (None, None)
        for bm in mim.brain_models:
            index_end = bm.index_offset + bm.index_count
            is_surface = bm.model_type == 'CIFTI_MODEL_TYPE_SURFACE'
            name.extend([bm.brain_structure] * bm.index_count)
            if is_surface:
                vertex[bm.index_offset:index_end] = bm.vertex_indices
                nvertices[bm.brain_structure] = bm.surface_number_of_vertices
            else:
                voxel[bm.index_offset:index_end, :] = bm.voxel_indices_ijk
                if affine is None:
                    shape = mim.volume.volume_dimensions
                    affine = mim.volume.transformation_matrix_voxel_indices_ijk_to_xyz.matrix
        return cls(name, voxel, vertex, affine, shape, nvertices)

    def to_mapping(self, dim):
        """
        Converts the brain model axis to a MatrixIndicesMap for storage in CIFTI-2 format

        Parameters
        ----------
        dim : int
            which dimension of the CIFTI-2 vector/matrix is described by this dataset (zero-based)

        Returns
        -------
        :class:`.cifti2.Cifti2MatrixIndicesMap`
        """
        mim = cifti2.Cifti2MatrixIndicesMap([dim], 'CIFTI_INDEX_TYPE_BRAIN_MODELS')
        for name, to_slice, bm in self.iter_structures():
            is_surface = name in self.nvertices.keys()
            if is_surface:
                voxels = None
                vertices = cifti2.Cifti2VertexIndices(bm.vertex)
                nvertex = self.nvertices[name]
            else:
                voxels = cifti2.Cifti2VoxelIndicesIJK(bm.voxel)
                vertices = None
                nvertex = None
                if mim.volume is None:
                    affine = cifti2.Cifti2TransformationMatrixVoxelIndicesIJKtoXYZ(-3, self.affine)
                    mim.volume = cifti2.Cifti2Volume(self.volume_shape, affine)
            cifti_bm = cifti2.Cifti2BrainModel(to_slice.start, len(bm), 'CIFTI_MODEL_TYPE_SURFACE' if is_surface else 'CIFTI_MODEL_TYPE_VOXELS', name, nvertex, voxels, vertices)
            mim.append(cifti_bm)
        return mim

    def iter_structures(self):
        """
        Iterates over all brain structures in the order that they appear along the axis

        Yields
        ------
        tuple with 3 elements:
        - CIFTI-2 brain structure name
        - slice to select the data associated with the brain structure from the tensor
        - brain model covering that specific brain structure
        """
        idx_start = 0
        start_name = self.name[idx_start]
        for idx_current, name in enumerate(self.name):
            if start_name != name:
                yield (start_name, slice(idx_start, idx_current), self[idx_start:idx_current])
                idx_start = idx_current
                start_name = self.name[idx_start]
        yield (start_name, slice(idx_start, None), self[idx_start:])

    @staticmethod
    def to_cifti_brain_structure_name(name):
        """
        Attempts to convert the name of an anatomical region in a format recognized by CIFTI-2

        This function returns:

        - the name if it is in the CIFTI-2 format already
        - if the name is a tuple the first element is assumed to be the structure name while
          the second is assumed to be the hemisphere (left, right or both). The latter will default
          to both.
        - names like left_cortex, cortex_left, LeftCortex, or CortexLeft will be converted to
          CIFTI_STRUCTURE_CORTEX_LEFT

        see :py:func:`nibabel.cifti2.tests.test_name` for examples of
        which conversions are possible

        Parameters
        ----------
        name: iterable of 2-element tuples of integer and string
            input name of an anatomical region

        Returns
        -------
        CIFTI-2 compatible name

        Raises
        ------
        ValueError: raised if the input name does not match a known anatomical structure in CIFTI-2
        """
        if name in cifti2.CIFTI_BRAIN_STRUCTURES:
            return cifti2.CIFTI_BRAIN_STRUCTURES.ciftiname[name]
        if not isinstance(name, str):
            if len(name) == 1:
                structure = name[0]
                orientation = 'both'
            else:
                structure, orientation = name
                if structure.lower() in ('left', 'right', 'both'):
                    orientation, structure = name
        else:
            orient_names = ('left', 'right', 'both')
            for poss_orient in orient_names:
                idx = len(poss_orient)
                if poss_orient == name.lower()[:idx]:
                    orientation = poss_orient
                    if name[idx] in '_ ':
                        structure = name[idx + 1:]
                    else:
                        structure = name[idx:]
                    break
                if poss_orient == name.lower()[-idx:]:
                    orientation = poss_orient
                    if name[-idx - 1] in '_ ':
                        structure = name[:-idx - 1]
                    else:
                        structure = name[:-idx]
                    break
            else:
                orientation = 'both'
                structure = name
        if orientation.lower() == 'both':
            proposed_name = f'CIFTI_STRUCTURE_{structure.upper()}'
        else:
            proposed_name = f'CIFTI_STRUCTURE_{structure.upper()}_{orientation.upper()}'
        if proposed_name not in cifti2.CIFTI_BRAIN_STRUCTURES.ciftiname:
            raise ValueError(f'{name} was interpreted as {proposed_name}, which is not a valid CIFTI brain structure')
        return proposed_name

    @property
    def surface_mask(self):
        """
        (N, ) boolean array which is true for any element on the surface
        """
        return np.vectorize(lambda name: name in self.nvertices.keys())(self.name)

    @property
    def volume_mask(self):
        """
        (N, ) boolean array which is true for any element on the surface
        """
        return np.vectorize(lambda name: name not in self.nvertices.keys())(self.name)
    _affine = None

    @property
    def affine(self):
        """
        Affine of the volumetric image in which the greyordinate voxels were defined
        """
        return self._affine

    @affine.setter
    def affine(self, value):
        if value is not None:
            value = np.asanyarray(value)
            if value.shape != (4, 4):
                raise ValueError('Affine transformation should be a 4x4 array')
        self._affine = value
    _volume_shape = None

    @property
    def volume_shape(self):
        """
        Shape of the volumetric image in which the greyordinate voxels were defined
        """
        return self._volume_shape

    @volume_shape.setter
    def volume_shape(self, value):
        if value is not None:
            value = tuple(value)
            if len(value) != 3:
                raise ValueError('Volume shape should be a tuple of length 3')
            if not all((isinstance(v, int) for v in value)):
                raise ValueError('All elements of the volume shape should be integers')
        self._volume_shape = value
    _name = None

    @property
    def name(self):
        """The brain structure to which the voxel/vertices of belong"""
        return self._name

    @name.setter
    def name(self, values):
        self._name = np.array([self.to_cifti_brain_structure_name(name) for name in values])

    def __len__(self):
        return self.name.size

    def __eq__(self, other):
        if not isinstance(other, BrainModelAxis) or len(self) != len(other):
            return False
        if xor(self.affine is None, other.affine is None):
            return False
        return (self.affine is None or (np.allclose(self.affine, other.affine) and self.volume_shape == other.volume_shape)) and self.nvertices == other.nvertices and np.array_equal(self.name, other.name) and np.array_equal(self.voxel[self.volume_mask], other.voxel[other.volume_mask]) and np.array_equal(self.vertex[self.surface_mask], other.vertex[other.surface_mask])

    def __add__(self, other):
        """
        Concatenates two BrainModels

        Parameters
        ----------
        other : BrainModelAxis
            brain model to be appended to the current one

        Returns
        -------
        BrainModelAxis
        """
        if not isinstance(other, BrainModelAxis):
            return NotImplemented
        if self.affine is None:
            affine, shape = (other.affine, other.volume_shape)
        else:
            affine, shape = (self.affine, self.volume_shape)
            if other.affine is not None and (not np.allclose(other.affine, affine) or other.volume_shape != shape):
                raise ValueError('Trying to concatenate two BrainModels defined in a different brain volume')
        nvertices = dict(self.nvertices)
        for name, value in other.nvertices.items():
            if name in nvertices.keys() and nvertices[name] != value:
                raise ValueError(f'Trying to concatenate two BrainModels with inconsistent number of vertices for {name}')
            nvertices[name] = value
        return self.__class__(np.append(self.name, other.name), np.concatenate((self.voxel, other.voxel), 0), np.append(self.vertex, other.vertex), affine, shape, nvertices)

    def __getitem__(self, item):
        """
        Extracts part of the brain structure

        Parameters
        ----------
        item : anything that can index a 1D array

        Returns
        -------
        If `item` is an integer returns a tuple with 3 elements:
        - boolean, which is True if it is a surface element
        - vertex index if it is a surface element, otherwise array with 3 voxel indices
        - structure.BrainStructure object describing the brain structure the element was taken from

        Otherwise returns a new BrainModelAxis
        """
        if isinstance(item, int):
            return self.get_element(item)
        if isinstance(item, str):
            raise IndexError('Can not index an Axis with a string (except for ParcelsAxis)')
        return self.__class__(self.name[item], self.voxel[item], self.vertex[item], self.affine, self.volume_shape, self.nvertices)

    def get_element(self, index):
        """
        Describes a single element from the axis

        Parameters
        ----------
        index : int
            Indexes the row/column of interest

        Returns
        -------
        tuple with 3 elements
        - str, 'CIFTI_MODEL_TYPE_SURFACE' for vertex or 'CIFTI_MODEL_TYPE_VOXELS' for voxel
        - vertex index if it is a surface element, otherwise array with 3 voxel indices
        - structure.BrainStructure object describing the brain structure the element was taken from
        """
        element_type = 'CIFTI_MODEL_TYPE_' + ('SURFACE' if self.name[index] in self.nvertices.keys() else 'VOXELS')
        struct = self.vertex if 'SURFACE' in element_type else self.voxel
        return (element_type, struct[index], self.name[index])