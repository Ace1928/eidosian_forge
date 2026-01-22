from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
def SaveAgentToFile(response, args):
    dest = args.destination
    if not IsBucketUri(dest):
        props = response.additionalProperties
        agent_content = next((prop for prop in props if prop.key == 'agentContent'))
        agent_content_bin = base64.b64decode(agent_content.value.string_value)
        log.WriteToFileOrStdout(dest, agent_content_bin, binary=True)
        if dest != '-':
            log.status.Print('Wrote agent to [{}].'.format(dest))
    return response