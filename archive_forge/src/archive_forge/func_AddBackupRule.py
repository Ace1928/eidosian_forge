from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.backupdr import util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddBackupRule(parser, required=True):
    """Adds a positional backup-rule argument to parser.

  Args:
    parser: argparse.Parser: Parser object for command line inputs.
    required: Whether or not --backup-rule is required.
  """
    rule_id_validator = arg_parsers.RegexpValidator('[a-z][a-z0-9-]{0,62}', 'Invalid rule-id. This human-readable name must be unique and start with a lowercase letter followed by up to 62 lowercase letters, numbers, or hyphens')
    backup_vault_validator = arg_parsers.RegexpValidator('projects\\/((?:[^:]+:)?[a-z0-9\\\\-]+)\\/locations\\/([a-zA-Z0-9-]+)\\/backupVaults\\/(.*)$', 'Invalid backup vault format. Expected format:\n      `projects/<project>/locations/<location>/backupVaults/<backup-vault-id>`')
    month_options = {'JAN': 'JANUARY', 'FEB': 'FEBRUARY', 'MAR': 'MARCH', 'APR': 'APRIL', 'MAY': 'MAY', 'JUN': 'JUNE', 'JUL': 'JULY', 'AUG': 'AUGUST', 'SEP': 'SEPTEMBER', 'OCT': 'OCTOBER', 'NOV': 'NOVEMBER', 'DEC': 'DECEMBER'}
    day_options = {'MON': 'MONDAY', 'TUE': 'TUESDAY', 'WED': 'WEDNESDAY', 'THU': 'THURSDAY', 'FRI': 'FRIDAY', 'SAT': 'SATURDAY', 'SUN': 'SUNDAY'}
    recurrence_options = ['HOURLY', 'DAILY', 'WEEKLY', 'MONTHLY', 'YEARLY']
    recurrence_validator = arg_parsers.CustomFunctionValidator(lambda arg: arg in recurrence_options, 'Recurrence should be one of the following: ' + ', '.join(recurrence_options), str)

    def ArgListParser(obj_parser, delim=' '):
        return arg_parsers.ArgList(obj_parser, custom_delim_char=delim)
    parser.add_argument('--backup-rule', required=required, type=arg_parsers.ArgDict(spec={'rule-id': rule_id_validator, 'backup-vault': backup_vault_validator, 'retention-days': int, 'recurrence': recurrence_validator, 'backup-window-start': arg_parsers.BoundedInt(0, 23), 'backup-window-end': arg_parsers.BoundedInt(1, 24), 'time-zone': str, 'hourly-frequency': arg_parsers.BoundedInt(6, 23), 'days-of-week': ArgListParser(util.OptionsMapValidator(day_options).Parse), 'days-of-month': ArgListParser(arg_parsers.BoundedInt(1, 31)), 'months': ArgListParser(util.OptionsMapValidator(month_options).Parse)}, required_keys=['rule-id', 'backup-vault', 'recurrence', 'retention-days', 'backup-window-start', 'backup-window-end']), action='append', metavar='PROPERTY=VALUE', help="\n          Name of the backup rule. A backup rule defines parameters for when and how a backup is created. This flag can be repeated to create more backup rules.\n\n          Parameters for the backup rule include::\n          - rule-id\n          - backup-vault\n          - retention-days\n          - recurrence\n          - backup-window-start\n          - backup-window-end\n          - time-zone\n\n          Along with any of these mutually exclusive flags:\n          - hourly-frequency (for HOURLY recurrence, expects value between 6-23)\n          - days-of-week (for WEEKLY recurrence, eg: 'MON TUE')\n          - days-of-month (for MONTHLY & YEARLY recurrence, eg: '1 7 5' days)\n          - months (for YEARLY recurrence, eg: 'JANUARY JUNE')\n\n          This flag can be repeated to specify multiple backup rules.\n\n          E.g., `rule-id=sample-daily-rule,backup-vault=projects/sample-project/locations/us-central1/backupVaults/sample-backup-vault,recurrence=WEEKLY,backup-window-start=2,backup-window-end=14,retention-days=20,days-of-week='SUNDAY MONDAY'`\n          ")