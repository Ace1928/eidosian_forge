from __future__ import (absolute_import, division, print_function)
from ansible import constants as C
from ansible import context
from ansible.playbook.attribute import FieldAttribute
from ansible.playbook.base import Base
from ansible.utils.display import Display

        Adds 'magic' variables relating to connections to the variable dictionary provided.
        In case users need to access from the play, this is a legacy from runner.
        