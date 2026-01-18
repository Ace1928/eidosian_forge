from the :meth:`.cifti2.Cifti2Header.get_axis` method on the header object
import abc
from operator import xor
import numpy as np
from . import cifti2
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