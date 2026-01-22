from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.command_lib.scc import errors
from googlecloudsdk.command_lib.scc import util
def ValidateAndGetBigQueryExportV1Name(args):
    """Returns relative resource name for a v1 B2igQuery export.

  Validates on regexes for args containing full names or short names with
  resources. Localization is supported by the
  ValidateAndGetBigQueryExportV2Name method.

  Args:
    args: an argparse object that should contain .BIG_QUERY_EXPORT, optionally 1
      of .organization, .folder, .project

  Examples:

  args with BIG_QUERY_EXPORT="organizations/123/bigQueryExports/config1"
  returns the BIG_QUERY_EXPORT

  args with BIG_QUERY_EXPORT="config1" and projects="projects/123" returns
  projects/123/bigQueryExports/config1
  """
    bq_export_name = args.BIG_QUERY_EXPORT
    long_name_format = re.compile('(organizations|projects|folders)/.*/bigQueryExports/[a-z]([a-z0-9-]{0,61}[a-z0-9])?$').match(bq_export_name)
    short_name_format = re.compile('^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$').match(bq_export_name)
    if not long_name_format and (not short_name_format):
        if '/' in bq_export_name:
            raise errors.InvalidSCCInputError('BigQuery export must match the full resource name, or `--organization=`, `--folder=` or `--project=` must be provided.')
        else:
            raise errors.InvalidSCCInputError("BigQuery export id does not match the pattern '^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$'.")
    if long_name_format:
        return bq_export_name
    if short_name_format:
        parent = util.GetParentFromNamedArguments(args)
        if parent is None:
            raise errors.InvalidSCCInputError('BigQuery export must match the full resource name, or `--organization=`, `--folder=` or `--project=` must be provided.')
        else:
            return util.GetParentFromNamedArguments(args) + '/bigQueryExports/' + bq_export_name