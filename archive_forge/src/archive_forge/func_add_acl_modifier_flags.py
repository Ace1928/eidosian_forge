from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def add_acl_modifier_flags(parser):
    """Adds flags common among commands that modify ACLs."""
    add_predefined_acl_flag(parser)
    parser.add_argument('--acl-file', help='Path to a local JSON or YAML formatted file containing a valid policy. See the [ObjectAccessControls resource](https://cloud.google.com/storage/docs/json_api/v1/objectAccessControls) for a representation of JSON formatted files. The output of `gcloud storage [buckets|objects] describe` `--format="multi(acl:format=json)"` is a valid file and can be edited for more fine-grained control.')
    parser.add_argument('--add-acl-grant', action='append', metavar='ACL_GRANT', type=arg_parsers.ArgDict(), help='Key-value pairs mirroring the JSON accepted by your cloud provider. For example, for Cloud Storage,`--add-acl-grant=entity=user-tim@gmail.com,role=OWNER`')
    parser.add_argument('--remove-acl-grant', action='append', help='Key-value pairs mirroring the JSON accepted by your cloud provider. For example, for Cloud Storage, `--remove-acl-grant=ENTITY`, where `ENTITY` has a valid ACL entity format, such as `user-tim@gmail.com`, `group-admins`, `allUsers`, etc.')