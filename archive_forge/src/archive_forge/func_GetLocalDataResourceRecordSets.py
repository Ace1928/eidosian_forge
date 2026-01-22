from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
import ipaddr
def GetLocalDataResourceRecordSets():
    return base.Argument('--local-data', type=arg_parsers.ArgDict(spec={'name': str, 'type': str, 'ttl': int, 'rrdatas': str}), metavar='LOCAL_DATA', action='append', help='    All resource record sets for this selector, one per resource record\n    type. The name must match the dns_name.\n\n    This is a repeated argument that can be specified multiple times to specify\n    multiple local data rrsets.\n    (e.g. --local-data=name="zone.com.",type="A",ttl=21600,rrdata="1.2.3.4 "\n    --local-data=name="www.zone.com.",type="CNAME",ttl=21600,rrdata="1.2.3.4|5.6.7.8")\n\n    *name*::: The DnsName of a resource record set.\n\n    *type*::: Type of all resource records in this set. For example, A, AAAA, SOA, MX,\n    NS, TXT ...\n\n    *ttl*::: Number of seconds that this ResourceRecordSet can be cached by resolvers.\n\n    *rrdatas*::: The list of datas for this record, split by "|".\n    ')