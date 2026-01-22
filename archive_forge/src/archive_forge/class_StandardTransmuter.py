from __future__ import annotations
import os
import re
from multiprocessing import Pool
from typing import TYPE_CHECKING, Callable
from pymatgen.alchemy.materials import TransformedStructure
from pymatgen.io.vasp.sets import MPRelaxSet, VaspInputSet
class StandardTransmuter:
    """An example of a Transmuter object, which performs a sequence of
    transformations on many structures to generate TransformedStructures.

    Attributes:
        transformed_structures (list[Structure]): List of all transformed structures.
    """

    def __init__(self, transformed_structures, transformations=None, extend_collection: int=0, ncores: int | None=None) -> None:
        """Initializes a transmuter from an initial list of
        pymatgen.alchemy.materials.TransformedStructure.

        Args:
            transformed_structures ([TransformedStructure]): Input transformed
                structures
            transformations ([Transformations]): New transformations to be
                applied to all structures.
            extend_collection (int): Whether to use more than one output
                structure from one-to-many transformations. extend_collection
                can be an int, which determines the maximum branching for each
                transformation.
            ncores (int): Number of cores to use for applying transformations.
                Uses multiprocessing.Pool. Default is None, which implies
                serial.
        """
        self.transformed_structures = transformed_structures
        self.ncores = ncores
        if transformations is not None:
            for trans in transformations:
                self.append_transformation(trans, extend_collection=extend_collection)

    def __getitem__(self, index):
        return self.transformed_structures[index]

    def __getattr__(self, name):
        return [getattr(x, name) for x in self.transformed_structures]

    def __len__(self):
        return len(self.transformed_structures)

    def __str__(self):
        output = ['Current structures', '------------']
        for x in self.transformed_structures:
            output.append(str(x.final_structure))
        return '\n'.join(output)

    def undo_last_change(self) -> None:
        """Undo the last transformation in the TransformedStructure.

        Raises:
            IndexError if already at the oldest change.
        """
        for x in self.transformed_structures:
            x.undo_last_change()

    def redo_next_change(self) -> None:
        """Redo the last undone transformation in the TransformedStructure.

        Raises:
            IndexError if already at the latest change.
        """
        for x in self.transformed_structures:
            x.redo_next_change()

    def append_transformation(self, transformation, extend_collection=False, clear_redo=True):
        """Appends a transformation to all TransformedStructures.

        Args:
            transformation: Transformation to append
            extend_collection: Whether to use more than one output structure
                from one-to-many transformations. extend_collection can be a
                number, which determines the maximum branching for each transformation.
            clear_redo (bool): Whether to clear the redo list. By default,
                this is True, meaning any appends clears the history of
                undoing. However, when using append_transformation to do a
                redo, the redo list should not be cleared to allow multiple redos.

        Returns:
            list[bool]: corresponding to initial transformed structures each boolean
                describes whether the transformation altered the structure
        """
        if self.ncores and transformation.use_multiprocessing:
            with Pool(self.ncores) as p:
                z = ((x, transformation, extend_collection, clear_redo) for x in self.transformed_structures)
                trafo_new_structs = p.map(_apply_transformation, z, 1)
                self.transformed_structures = []
                for ts in trafo_new_structs:
                    self.transformed_structures.extend(ts)
        else:
            new_structures = []
            for x in self.transformed_structures:
                new = x.append_transformation(transformation, extend_collection, clear_redo=clear_redo)
                if new is not None:
                    new_structures.extend(new)
            self.transformed_structures.extend(new_structures)

    def extend_transformations(self, transformations):
        """Extends a sequence of transformations to the TransformedStructure.

        Args:
            transformations: Sequence of Transformations
        """
        for trafo in transformations:
            self.append_transformation(trafo)

    def apply_filter(self, structure_filter):
        """Applies a structure_filter to the list of TransformedStructures
        in the transmuter.

        Args:
            structure_filter: StructureFilter to apply.
        """

        def test_transformed_structure(ts):
            return structure_filter.test(ts.final_structure)
        self.transformed_structures = list(filter(test_transformed_structure, self.transformed_structures))
        for ts in self.transformed_structures:
            ts.append_filter(structure_filter)

    def write_vasp_input(self, **kwargs):
        """Batch write vasp input for a sequence of transformed structures to
        output_dir, following the format output_dir/{formula}_{number}.

        Args:
            kwargs: All kwargs supported by batch_write_vasp_input.
        """
        batch_write_vasp_input(self.transformed_structures, **kwargs)

    def set_parameter(self, key, value):
        """Add parameters to the transmuter. Additional parameters are stored in
        the as_dict() output.

        Args:
            key: The key for the parameter.
            value: The value for the parameter.
        """
        for x in self.transformed_structures:
            x.other_parameters[key] = value

    def add_tags(self, tags):
        """Add tags for the structures generated by the transmuter.

        Args:
            tags: A sequence of tags. Note that this should be a sequence of
                strings, e.g., ["My awesome structures", "Project X"].
        """
        self.set_parameter('tags', tags)

    def append_transformed_structures(self, trafo_structs_or_transmuter):
        """Method is overloaded to accept either a list of transformed structures
        or transmuter, it which case it appends the second transmuter"s
        structures.

        Args:
            trafo_structs_or_transmuter: A list of transformed structures or a
                transmuter.
        """
        if isinstance(trafo_structs_or_transmuter, self.__class__):
            self.transformed_structures.extend(trafo_structs_or_transmuter.transformed_structures)
        else:
            for ts in trafo_structs_or_transmuter:
                assert isinstance(ts, TransformedStructure)
            self.transformed_structures.extend(trafo_structs_or_transmuter)

    @classmethod
    def from_structures(cls, structures, transformations=None, extend_collection=0) -> Self:
        """Alternative constructor from structures rather than
        TransformedStructures.

        Args:
            structures: Sequence of structures
            transformations: New transformations to be applied to all
                structures
            extend_collection: Whether to use more than one output structure
                from one-to-many transformations. extend_collection can be a
                number, which determines the maximum branching for each
                transformation.

        Returns:
            StandardTransmuter
        """
        trafo_struct = [TransformedStructure(s, []) for s in structures]
        return cls(trafo_struct, transformations, extend_collection)