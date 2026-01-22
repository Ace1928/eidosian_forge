from __future__ import annotations
import warnings
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import requires
from pymatgen.core.tensors import Tensor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
class InternalStrainTensor:
    """
    This class describes the Nx3x3x3 internal tensor defined by a
    structure, point operations of the structure's atomic sites.
    """

    def __init__(self, structure: Structure, ist, pointops, tol: float=0.001):
        """
        Create an InternalStrainTensor object.

        Args:
            input_matrix (Nx3x3x3 array-like): the Nx3x3x3 array-like
                representing the internal strain tensor
        """
        self.structure = structure
        self.ist = ist
        self.pointops = pointops
        self.IST_operations = None
        obj = self.ist
        if not (obj - np.transpose(obj, (0, 1, 3, 2)) < tol).all():
            warnings.warn('Input internal strain tensor does not satisfy standard symmetries')

    def get_IST_operations(self, opstol=0.001):
        """
        Returns the symmetry operations which maps the tensors
        belonging to equivalent sites onto each other in the form
        [site index 1, site index 2, [Symmops mapping from site
        index 1 to site index 2]].

        Args:
            opstol (float): tolerance for determining if a symmetry
            operation relates two sites

        Returns:
            list of symmetry operations mapping equivalent sites and
            the indexes of those sites.
        """
        struct = self.structure
        ops = SpacegroupAnalyzer(struct).get_symmetry_operations(cartesian=True)
        uniq_point_ops = list(ops)
        for ops in self.pointops:
            for op in ops:
                if op not in uniq_point_ops:
                    uniq_point_ops.append(op)
        IST_operations = []
        for atom in range(len(self.ist)):
            IST_operations.append([])
            for j in range(atom):
                for op in uniq_point_ops:
                    new = op.transform_tensor(self.ist[j])
                    if np.allclose(new, self.ist[atom], atol=opstol):
                        IST_operations[atom].append([j, op])
        self.IST_operations = IST_operations

    def get_rand_IST(self, max_force=1):
        """
        Generate a random internal strain tensor which obeys a structure's
        symmetry and the acoustic sum rule.

        Args:
            max_force (float): maximum born effective charge value

        Returns:
            InternalStrainTensor
        """
        n_atoms = len(self.structure)
        IST = np.zeros((n_atoms, 3, 3, 3))
        for atom, ops in enumerate(self.IST_operations):
            temp_tensor = np.zeros([3, 3, 3])
            for op in ops:
                temp_tensor += op[1].transform_tensor(IST[op[0]])
            if len(ops) == 0:
                temp_tensor = Tensor(np.random.rand(3, 3, 3) - 0.5)
                for dim in range(3):
                    temp_tensor[dim] = (temp_tensor[dim] + temp_tensor[dim].T) / 2
                temp_tensor = sum((temp_tensor.transform(symm_op) for symm_op in self.pointops[atom])) / len(self.pointops[atom])
            IST[atom] = temp_tensor
            if len(ops) != 0:
                IST[atom] = IST[atom] / len(ops)
        return IST * max_force