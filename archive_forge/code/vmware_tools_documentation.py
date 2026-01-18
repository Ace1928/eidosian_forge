from __future__ import absolute_import, division, print_function
import re
from os.path import exists, getsize
from socket import gaierror
from ssl import SSLError
from time import sleep
import traceback
from ansible.errors import AnsibleError, AnsibleFileNotFound, AnsibleConnectionFailure
from ansible.module_utils._text import to_bytes, to_native
from ansible.plugins.connection import ConnectionBase
from ansible.module_utils.basic import missing_required_lib

            we need to warp the execution of powershell into a cmd /c because
            the call otherwise fails with "Authentication or permission failure"
            #FIXME: Fix the unecessary invocation of cmd and run the command directly
            