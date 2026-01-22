from the :meth:`.cifti2.Cifti2Header.get_axis` method on the header object
import abc
from operator import xor
import numpy as np
from . import cifti2
class ParcelsAxis(Axis):
    """
    Each row/column in the CIFTI-2 vector/matrix represents a parcel of voxels/vertices

    This Axis describes which parcel is represented by each row/column.

    Individual parcels can be accessed based on their name, using
    ``parcel = parcel_axis[name]``
    """

    def __init__(self, name, voxels, vertices, affine=None, volume_shape=None, nvertices=None):
        """
        Use of this constructor is not recommended. New ParcelsAxis axes can be constructed more
        easily from a sequence of BrainModelAxis axes using
        :py:meth:`~ParcelsAxis.from_brain_models`

        Parameters
        ----------
        name : array_like
            (N, ) string array with the parcel names
        voxels :  array_like
            (N, ) object array each containing a sequence of voxels.
            For each parcel the voxels are represented by a (M, 3) index array
        vertices :  array_like
            (N, ) object array each containing a sequence of vertices.
            For each parcel the vertices are represented by a mapping from brain structure name to
            (M, ) index array
        affine : array_like, optional
            (4, 4) array mapping voxel indices to mm space (not needed for CIFTI-2 files only
            covering the surface)
        volume_shape : tuple of three integers, optional
            shape of the volume in which the voxels were defined (not needed for CIFTI-2 files only
            covering the surface)
        nvertices : dict from string to integer, optional
            maps names of surface elements to integers (not needed for volumetric CIFTI-2 files)
        """
        self.name = np.asanyarray(name, dtype='U')
        self.voxels = np.empty(len(voxels), dtype='object')
        for idx, vox in enumerate(voxels):
            self.voxels[idx] = vox
        self.vertices = np.asanyarray(vertices, dtype='object')
        self.affine = np.asanyarray(affine) if affine is not None else None
        self.volume_shape = volume_shape
        if nvertices is None:
            self.nvertices = {}
        else:
            self.nvertices = {BrainModelAxis.to_cifti_brain_structure_name(name): number for name, number in nvertices.items()}
        for check_name in ('name', 'voxels', 'vertices'):
            if getattr(self, check_name).shape != (self.size,):
                raise ValueError(f'Input {check_name} has incorrect shape ({getattr(self, check_name).shape}) for Parcel axis')

    @classmethod
    def from_brain_models(cls, named_brain_models):
        """
        Creates a Parcel axis from a list of BrainModelAxis axes with names

        Parameters
        ----------
        named_brain_models : iterable of 2-element tuples of string and BrainModelAxis
            list of (parcel name, brain model representation) pairs defining each parcel

        Returns
        -------
        ParcelsAxis
        """
        nparcels = len(named_brain_models)
        affine = None
        volume_shape = None
        all_names = []
        all_voxels = np.zeros(nparcels, dtype='object')
        all_vertices = np.zeros(nparcels, dtype='object')
        nvertices = {}
        for idx_parcel, (parcel_name, bm) in enumerate(named_brain_models):
            all_names.append(parcel_name)
            voxels = bm.voxel[bm.volume_mask]
            if voxels.shape[0] != 0:
                if affine is None:
                    affine = bm.affine
                    volume_shape = bm.volume_shape
                elif not np.allclose(affine, bm.affine) or volume_shape != bm.volume_shape:
                    raise ValueError('Can not combine brain models defined in different volumes into a single Parcel axis')
            all_voxels[idx_parcel] = voxels
            vertices = {}
            for name, _, bm_part in bm.iter_structures():
                if name in bm.nvertices.keys():
                    if name in nvertices.keys() and nvertices[name] != bm.nvertices[name]:
                        raise ValueError(f'Got multiple conflicting number of vertices for surface structure {name}')
                    nvertices[name] = bm.nvertices[name]
                    vertices[name] = bm_part.vertex
            all_vertices[idx_parcel] = vertices
        return ParcelsAxis(all_names, all_voxels, all_vertices, affine, volume_shape, nvertices)

    @classmethod
    def from_index_mapping(cls, mim):
        """
        Creates a new Parcels axis based on a CIFTI-2 dataset

        Parameters
        ----------
        mim : :class:`cifti2.Cifti2MatrixIndicesMap`

        Returns
        -------
        ParcelsAxis
        """
        nparcels = len(list(mim.parcels))
        all_names = []
        all_voxels = np.zeros(nparcels, dtype='object')
        all_vertices = np.zeros(nparcels, dtype='object')
        volume_shape = None if mim.volume is None else mim.volume.volume_dimensions
        affine = None
        if mim.volume is not None:
            affine = mim.volume.transformation_matrix_voxel_indices_ijk_to_xyz.matrix
        nvertices = {}
        for surface in mim.surfaces:
            nvertices[surface.brain_structure] = surface.surface_number_of_vertices
        for idx_parcel, parcel in enumerate(mim.parcels):
            nvoxels = 0 if parcel.voxel_indices_ijk is None else len(parcel.voxel_indices_ijk)
            voxels = np.zeros((nvoxels, 3), dtype='i4')
            if nvoxels != 0:
                voxels[:] = parcel.voxel_indices_ijk
            vertices = {}
            for vertex in parcel.vertices:
                name = vertex.brain_structure
                vertices[vertex.brain_structure] = np.array(vertex)
                if name not in nvertices.keys():
                    raise ValueError(f'Number of vertices for surface structure {name} not defined')
            all_voxels[idx_parcel] = voxels
            all_vertices[idx_parcel] = vertices
            all_names.append(parcel.name)
        return cls(all_names, all_voxels, all_vertices, affine, volume_shape, nvertices)

    def to_mapping(self, dim):
        """
        Converts the Parcel to a MatrixIndicesMap for storage in CIFTI-2 format

        Parameters
        ----------
        dim : int
            which dimension of the CIFTI-2 vector/matrix is described by this dataset (zero-based)

        Returns
        -------
        :class:`cifti2.Cifti2MatrixIndicesMap`
        """
        mim = cifti2.Cifti2MatrixIndicesMap([dim], 'CIFTI_INDEX_TYPE_PARCELS')
        if self.affine is not None:
            affine = cifti2.Cifti2TransformationMatrixVoxelIndicesIJKtoXYZ(-3, matrix=self.affine)
            mim.volume = cifti2.Cifti2Volume(self.volume_shape, affine)
        for name, nvertex in self.nvertices.items():
            mim.append(cifti2.Cifti2Surface(name, nvertex))
        for name, voxels, vertices in zip(self.name, self.voxels, self.vertices):
            cifti_voxels = cifti2.Cifti2VoxelIndicesIJK(voxels)
            element = cifti2.Cifti2Parcel(name, cifti_voxels)
            for name_vertex, idx_vertices in vertices.items():
                element.vertices.append(cifti2.Cifti2Vertices(name_vertex, idx_vertices))
            mim.append(element)
        return mim
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

    def __len__(self):
        return self.name.size

    def __eq__(self, other):
        if self.__class__ != other.__class__ or len(self) != len(other) or (not np.array_equal(self.name, other.name)) or (self.nvertices != other.nvertices) or any((not np.array_equal(vox1, vox2) for vox1, vox2 in zip(self.voxels, other.voxels))):
            return False
        if self.affine is not None:
            if other.affine is None or not np.allclose(self.affine, other.affine) or self.volume_shape != other.volume_shape:
                return False
        elif other.affine is not None:
            return False
        for vert1, vert2 in zip(self.vertices, other.vertices):
            if len(vert1) != len(vert2):
                return False
            for name in vert1.keys():
                if name not in vert2 or not np.array_equal(vert1[name], vert2[name]):
                    return False
        return True

    def __add__(self, other):
        """
        Concatenates two Parcels

        Parameters
        ----------
        other : ParcelsAxis
            parcel to be appended to the current one

        Returns
        -------
        Parcel
        """
        if not isinstance(other, ParcelsAxis):
            return NotImplemented
        if self.affine is None:
            affine, shape = (other.affine, other.volume_shape)
        else:
            affine, shape = (self.affine, self.volume_shape)
            if other.affine is not None and (not np.allclose(other.affine, affine) or other.volume_shape != shape):
                raise ValueError('Trying to concatenate two ParcelsAxis defined in a different brain volume')
        nvertices = dict(self.nvertices)
        for name, value in other.nvertices.items():
            if name in nvertices.keys() and nvertices[name] != value:
                raise ValueError(f'Trying to concatenate two ParcelsAxis with inconsistent number of vertices for {name}')
            nvertices[name] = value
        return self.__class__(np.append(self.name, other.name), np.append(self.voxels, other.voxels), np.append(self.vertices, other.vertices), affine, shape, nvertices)

    def __getitem__(self, item):
        """
        Extracts subset of the axes based on the type of ``item``:

        - `int`: 3-element tuple of (parcel name, parcel voxels, parcel vertices)
        - `string`: 2-element tuple of (parcel voxels, parcel vertices
        - other object that can index 1D arrays: new Parcel axis
        """
        if isinstance(item, str):
            idx = np.where(self.name == item)[0]
            if len(idx) == 0:
                raise IndexError(f'Parcel {item} not found')
            if len(idx) > 1:
                raise IndexError(f'Multiple parcels with name {item} found')
            return (self.voxels[idx[0]], self.vertices[idx[0]])
        if isinstance(item, int):
            return self.get_element(item)
        return self.__class__(self.name[item], self.voxels[item], self.vertices[item], self.affine, self.volume_shape, self.nvertices)

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
        - unicode name of the parcel
        - (M, 3) int array with voxel indices
        - dict from string to (K, ) int array with vertex indices
          for a specific surface brain structure
        """
        return (self.name[index], self.voxels[index], self.vertices[index])