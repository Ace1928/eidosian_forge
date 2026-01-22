import json
import sys
import textwrap
import traceback
import enum
from oslo_config import cfg
import prettytable
from oslo_upgradecheck._i18n import _
def _get_details(self, upgrade_check_result):
    if upgrade_check_result.details is not None:
        return '\n'.join(textwrap.wrap(upgrade_check_result.details, 60, subsequent_indent='  '))