import io
import os
import pathlib
from typing import TYPE_CHECKING, Optional, Sequence, Type, Union
from wandb import util
from wandb.sdk.lib import runid
from wandb.sdk.lib.paths import LogicalPath
from ._private import MEDIA_TMP
from .base_types.media import BatchableMedia, Media
Convert SMILES string to wandb.Molecule.

        Arguments:
            data: (string)
                SMILES string.
            caption: (string)
                Caption associated with the molecule for display
            sanitize: (bool)
                Check if the molecule is chemically reasonable by the RDKit's definition.
            convert_to_3d_and_optimize: (bool)
                Convert to rdkit.Chem.rdchem.Mol with 3D coordinates.
                This is an expensive operation that may take a long time for complicated molecules.
            mmff_optimize_molecule_max_iterations: (int)
                Number of iterations to use in rdkit.Chem.AllChem.MMFFOptimizeMolecule
        