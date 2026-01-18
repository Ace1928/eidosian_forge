import json
import sys
import textwrap
import traceback
import enum
from oslo_config import cfg
import prettytable
from oslo_upgradecheck._i18n import _
Performs checks to see if the deployment is ready for upgrade.

        These checks are expected to be run BEFORE services are restarted with
        new code.

        :returns: Code
        