from __future__ import (absolute_import, division, print_function)
import ansible.constants as C
from ansible.errors import AnsibleParserError
from ansible.playbook.block import Block
from ansible.playbook.task import Task
from ansible.utils.display import Display
from ansible.utils.sentinel import Sentinel

        This method is used to create the parent block for the included tasks
        when ``apply`` is specified
        