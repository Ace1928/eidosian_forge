from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.scc import securitycenter_client
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.scc import flags as scc_flags
from googlecloudsdk.command_lib.scc import util as scc_util
from googlecloudsdk.command_lib.scc.findings import flags
from googlecloudsdk.command_lib.scc.findings import util
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import times
@base.ReleaseTracks(base.ReleaseTrack.GA, base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class UpdateMarks(base.UpdateCommand):
    """Update Security Command Center finding's security marks."""
    detailed_help = {'DESCRIPTION': "Update Security Command Center finding's security marks.", 'EXAMPLES': '\n      Selectively update security mark `Key1` with value `v1` on testFinding. Note\n      that other security marks on `testFinding` are untouched:\n\n        $ {command} `testFinding` --organization=123456 --source=5678\n          --security-marks="key1=v1" --update-mask="marks.markKey1"\n\n      Update all security marks on `testFinding`, under project `example-project`\n      and source `5678`:\n\n        $ {command} projects/example-project/sources/5678/findings/testFinding\n          --security-marks="key1=v1,key2=v2"\n\n      Update all security marks on `testFinding`, under folder `456` and source\n      `5678`:\n\n        $ {command} folders/456/sources/5678/findings/testFinding\n          --security-marks="key1=v1,key2=v2"\n\n      Update all security marks on `testFinding`, under organization `123456` and\n      source `5678`:\n\n        $ {command} `testFinding` --organization=123456 --source=5678\n          --security-marks="key1=v1,key2=v2"\n\n      Delete all security marks on `testFinding`:\n\n        $ {command} `testFinding` --organization=123456 --source=5678\n          --security-marks=""\n\n      Update all security marks on `testFinding`, under project `example-project`,\n      source `5678` and `location=eu`:\n\n        $ {command} projects/example-project/sources/5678/findings/testFinding\n          --security-marks="key1=v1,key2=v2" --location=eu', 'API REFERENCE': '\n      This command uses the Security Command Center API. For more information,\n      see [Security Command Center API.](https://cloud.google.com/security-command-center/docs/reference/rest)'}

    @staticmethod
    def Args(parser):
        flags.CreateFindingArg().AddToParser(parser)
        scc_flags.API_VERSION_FLAG.AddToParser(parser)
        scc_flags.LOCATION_FLAG.AddToParser(parser)
        base.Argument('--security-marks', help='\n        SecurityMarks resource to be passed as the request body. It\'s a\n        key=value pair separated by comma (,). For example:\n        --security-marks="key1=val1,key2=val2".', type=arg_parsers.ArgDict(), metavar='KEY=VALUE').AddToParser(parser)
        parser.add_argument('--start-time', help='\n        Time at which the updated SecurityMarks take effect. See `$ gcloud topic\n        datetimes` for information on supported time formats.')
        parser.add_argument('--update-mask', help='\n        Use update-mask if you want to selectively update marks represented by\n        --security-marks flag. For example:\n        --update-mask="marks.key1,marks.key2". If you want to override all the\n        marks for the given finding either skip the update-mask flag or provide\n        an empty value (--update-mask \'\') for it.')
        parser.display_info.AddFormat(properties.VALUES.core.default_format.Get())

    def Run(self, args):
        version = _GetApiVersion(args)
        messages = securitycenter_client.GetMessages(version)
        request = messages.SecuritycenterOrganizationsSourcesFindingsUpdateSecurityMarksRequest()
        if args.start_time:
            start_time_dt = times.ParseDateTime(args.start_time)
            request.startTime = times.FormatDateTime(start_time_dt)
        client = securitycenter_client.GetClient(version)
        request.updateMask = args.update_mask
        if version == 'v1':
            security_marks = messages.SecurityMarks()
            security_marks.marks = encoding.DictToMessage(args.security_marks, messages.SecurityMarks.MarksValue)
            request.securityMarks = security_marks
        elif version == 'v2':
            security_marks = messages.GoogleCloudSecuritycenterV2SecurityMarks()
            security_marks.marks = encoding.DictToMessage(args.security_marks, messages.GoogleCloudSecuritycenterV2SecurityMarks.MarksValue)
            request.googleCloudSecuritycenterV2SecurityMarks = security_marks
        request = _ValidateParentAndUpdateName(args, request, version)
        if request.updateMask is not None:
            request.updateMask = scc_util.CleanUpUserMaskInput(request.updateMask)
        marks = client.organizations_sources_findings.UpdateSecurityMarks(request)
        return marks