from collections import OrderedDict, defaultdict
import os
import os.path as op
from pathlib import Path
import shutil
import socket
from copy import deepcopy
from glob import glob
from logging import INFO
from tempfile import mkdtemp
from ... import config, logging
from ...utils.misc import flatten, unflatten, str2bool, dict_diff
from ...utils.filemanip import (
from ...interfaces.base import (
from ...interfaces.base.specs import get_filecopy_info
from .utils import (
from .base import EngineBase
def _add_join_item_fields(self):
    """Add new join item fields assigned to the next iterated
        input

        This method is intended solely for workflow graph expansion.

        Examples
        --------

        >>> from nipype.interfaces.utility import IdentityInterface
        >>> import nipype.pipeline.engine as pe
        >>> from nipype import Node, JoinNode, Workflow
        >>> inputspec = Node(IdentityInterface(fields=['image']),
        ...    name='inputspec'),
        >>> join = JoinNode(IdentityInterface(fields=['images', 'mask']),
        ...    joinsource='inputspec', joinfield='images', name='join')
        >>> join._add_join_item_fields()
        {'images': 'imagesJ1'}

        Return the {base field: slot field} dictionary
        """
    idx = self._next_slot_index
    newfields = dict([(field, self._add_join_item_field(field, idx)) for field in self.joinfield])
    logger.debug('Added the %s join item fields %s.', self, newfields)
    self._next_slot_index += 1
    return newfields