import string
import textwrap
from ansible.module_utils._text import to_native
from ansible.module_utils.six.moves.urllib import parse as urlparse
def _linux_callback_script(tower_address, template_id, host_config_key):
    template_id = urlparse.quote(template_id)
    tower_address = urlparse.quote(tower_address)
    host_config_key = host_config_key.replace("'", '\'"\'"\'')
    script_tpl = '    #!/bin/bash\n    set -x\n\n    retry_attempts=10\n    attempt=0\n    while [[ $attempt -lt $retry_attempts ]]\n    do\n      status_code=$(curl --max-time 10 -v -k -s -i         --data \'host_config_key=${host_config_key}\'         \'https://${tower_address}/api/v2/job_templates/${template_id}/callback/\'         | head -n 1         | awk \'{print $2}\')\n      if [[ $status_code == 404 ]]\n        then\n        status_code=$(curl --max-time 10 -v -k -s -i           --data \'host_config_key=${host_config_key}\'           \'https://${tower_address}/api/v1/job_templates/${template_id}/callback/\'           | head -n 1           | awk \'{print $2}\')\n        # fall back to using V1 API for Tower 3.1 and below, since v2 API will always 404\n      fi\n      if [[ $status_code == 201 ]]\n        then\n        exit 0\n      fi\n      attempt=$(( attempt + 1 ))\n      echo "$${status_code} received... retrying in 1 minute. (Attempt $${attempt})"\n      sleep 60\n    done\n    exit 1\n    '
    tpl = string.Template(textwrap.dedent(script_tpl))
    return tpl.safe_substitute(tower_address=tower_address, template_id=template_id, host_config_key=host_config_key)