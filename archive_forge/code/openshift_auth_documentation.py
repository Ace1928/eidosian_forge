from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib_parse import urlparse, parse_qs, urlencode
from urllib.parse import urljoin
from base64 import urlsafe_b64encode
import hashlib

      openshift convert the access token to an OAuthAccessToken resource name using the algorithm
      https://github.com/openshift/console/blob/9f352ba49f82ad693a72d0d35709961428b43b93/pkg/server/server.go#L609-L613
    